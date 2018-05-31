from keras.layers import Embedding
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.cross_validation import train_test_split
from keras.models import load_model
import numpy as np
import os
import pandas as pd
from nltk.corpus import stopwords
from sklearn.utils import shuffle

def to_vector(sentence_list):
    MAX_TOKEN_NUM_WORDS = 2000
    MAX_SEQUENCES_LEN = 1000

    tokenizer = Tokenizer(num_words=MAX_TOKEN_NUM_WORDS)
    tokenizer.fit_on_texts(sentence_list)
    sequences = tokenizer.texts_to_sequences(sentence_list)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    sentence_vector = pad_sequences(sequences, maxlen=MAX_SEQUENCES_LEN)
    return sentence_vector

def clean_data(data_pandas):
    data_pandas.loc[data_pandas.label == 4, 'label'] = 1
    X = np.array(data_pandas['sentence'])
    y = np.array(data_pandas['label'])
    return X, y


def read_data(frac=1.0):
    FILE_PATH = 'data'
    FILE_NAME = 'emotion_data.csv'

    data_pandas = pd.read_csv(os.path.join(FILE_PATH, FILE_NAME), \
        names=['label','time_1','time_2','if_query','if_special','sentence'], \
        usecols=['label', 'sentence'])
    data_pandas = shuffle(data_pandas)
    data_pandas = data_pandas.sample(frac=frac)
    return data_pandas 

def split_data(X, y, *, train_size=0.99, val_size=0.008):
    test_size = 1 - train_size - val_size
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, train_size=train_size)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, train_size=val_size/(val_size + test_size))
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(x_train, y_train, x_test, y_test):
    MAX_FEATURES = 2000
    BATCH_SIZE = 32
    EPOCHS = 16
    print('Build model...')
    model = Sequential()
    model.add(Embedding(MAX_FEATURES, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=BATCH_SIZE)
    print('Test score:', score)
    print('Test accuracy:', acc)
    return model


def predict(x):
    model = load_model('model/test_model')
    y_hat = model.predict(x).reshape(x.shape[0])
    y_hat_binary = np.array(list(map(lambda x: 1 if (x > 0.5) else 0, y_hat)))
    return y_hat_binary

def save_test_result(X_test, y_test, y_hat):
    test_data = pd.DataFrame({'x_test': X_test, 'y_test': y_test, 'y_hat': y_hat})
    test_data.to_csv('test/test_result', index=False)

def calculate_acc():
    result_df = pd.read_csv('test/test_result')
    result_df['if_equals'] = result_df.y_test == result_df.y_hat
    equals_num = result_df[result_df.if_equals == True].count(numeric_only=True)['if_equals']
    all_num = result_df.count(numeric_only=True)['if_equals']
    return equals_num / all_num


def main():
    # lable:1 = positive label:0 = negavite
    data_df = read_data(0.05)
    X, y = clean_data(data_df)
    X_train_text, X_val_text, X_test_text, y_train, y_val, y_test = split_data(X, y)
    X_train, X_val, X_test = to_vector(X_train_text), to_vector(X_val_text), to_vector(X_test_text)
    model = build_model(X_train, y_train, X_val, y_val)
    model.save('model/test_model')

    y_hat = predict(X_test)
    save_test_result(X_test_text, y_test, y_hat)
    rate = calculate_acc()
    print(rate)


if __name__ == '__main__':
    main()