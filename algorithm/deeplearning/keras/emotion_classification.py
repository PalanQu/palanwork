from keras.layers import Embedding
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.cross_validation import train_test_split
import numpy as np
import os
import pandas as pd
from nltk.corpus import stopwords
from sklearn.utils import shuffle

"""
    tokenizer, lemma, remove stop words, to_lowcase, to_vector
"""



def to_vector(sentence_list):
    
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(sentence_list)
    sequences = tokenizer.texts_to_sequences(sentence_list)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    sentence_vector = pad_sequences(sequences, maxlen=50)
    return sentence_vector



def read_data():
    FILE_PATH = 'data'
    FILE_NAME = 'emotion_data.csv'

    data_pandas = pd.read_csv(os.path.join(FILE_PATH, FILE_NAME), \
        names=['label','time_1','time_2','if_query','if_special','sentence'], \
        usecols=['label', 'sentence'])
    data_pandas.loc[data_pandas.label == 4, 'label'] = 1

    # nagivate_sample = data_pandas.loc[data_pandas.label == 0].sample(n=10)
    # positive_sample = data_pandas.loc[data_pandas.label == 1].sample(n=10)
    samples = data_pandas.sample(n=2000)

    data_pandas.drop(index=samples.index, inplace=True)
    data_pandas = shuffle(data_pandas)
    data_pandas = data_pandas[:500000]

    sentence_list_train = np.array(data_pandas['sentence'])
    label_list_train = np.array(data_pandas['label'])
    sentence_list_sample = np.array(samples['sentence'])
    label_list_sample = np.array(samples['label'])

    return sentence_list_train, sentence_list_sample, label_list_train, label_list_sample 

def build_model(x_train, y_train, x_test, y_test):
    max_features = 1000 
    batch_size = 32
    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=8,
            validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    model.save('text_model')



def main():
    sentence_list_train, sentence_list_sample, label_list_train, label_list_sample = read_data()
    sentence_vector_train = to_vector(sentence_list_train)
    x_train, x_val, y_train, y_val =  train_test_split(sentence_vector_train, label_list_train, test_size=0.01)
    # sentence_vector_sample = to_vector(sentence_list_sample)
    # x_train, x_val, y_train, y_val =  train_test_split(sentence_vector_sample, label_list_sample, test_size=0.3)

    build_model(x_train, y_train, x_val, y_val)

if __name__ == '__main__':
    main()