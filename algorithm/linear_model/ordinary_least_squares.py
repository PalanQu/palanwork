import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression


def standard_lr(x, y):
    model = LinearRegression()
    model.fit(x, y)

def gradient_descent(x, y):
    m = x.shape[0]
    x = np.insert(x, 0, 1, axis=1)
    paras_init = np.full(x.shape[1], 1)
    updated_paras = paras_init
    alpha = 0.1
    iterator_max_num = 50
    accept_error = 1000

    for iter_num in range(iterator_max_num):
        descent_list = np.dot(np.dot(x, updated_paras) - y, x) / m
        updated_paras = updated_paras - alpha * descent_list
        error = np.sum(np.square(np.dot(x, updated_paras) - y))
        if (error < accept_error):
            break
        print(iter_num, "\t" , error)    

def normal_equation(x, y):
    x = np.mat(np.insert(x, 0, 1, axis=1))
    y = np.mat(y.reshape(y.shape[0],1))

    paras_list = (x.T @ x).I @ x.T @ y
    print(paras_list)

def main():
    x, y, coef = make_regression(n_samples=200, n_features=5, n_informative=50 ,random_state=0, coef=True)
    normal_equation(x, y)
    # gradient_descent(x, y)
    standard_lr(x, y)

if __name__ == '__main__':
    main()