import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

def standard_lr(x, y):
    model = LogisticRegression(fit_intercept=False)
    model.fit(x, y)
    print(model.coef_)

def gradient_descent_without_regulatization(x, y, gamma):
    m = x.shape[0]
    paras_init = np.full(x.shape[1], 1)
    updated_paras = paras_init
    alpha = 0.1
    iterator_max_num = 500
    accept_error = 1

    for iter_num in range(iterator_max_num):
        # descent_list = ((1 / (1 + np.exp(-(x @ updated_paras))) - y) @ x + gamma * updated_paras) / m
        descent_list = ((sigmoid(x @ updated_paras) - y) @ x + gamma * updated_paras) / m
        updated_paras = updated_paras - alpha * descent_list
        error = np.sum(np.square(np.dot(x, updated_paras) - y))
        if (error < accept_error):
            break
        # print(iter_num, "\t" , error)    
    
    print(updated_paras)


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def logistic_regression_test(features, target):
    weights = np.zeros(features.shape[1])
    num_steps = 500
    learning_rate = 0.1
    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient
        
        # Print log-likelihood every so often
        # if step % 10000 == 0:
        #     print (log_likelihood(features, target, weights))
    print(weights)
        
    
    # return weights


def main():
    x, y = make_classification(n_samples=100, n_features=5, random_state=0)
    standard_lr(x, y)
    gradient_descent_without_regulatization(x, y, 1)
    logistic_regression_test(x, y)

if __name__ == '__main__':
    main()