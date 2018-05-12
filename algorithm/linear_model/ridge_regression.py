import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

def standard_ridge(x, y, gamma):
    model = Ridge(alpha=1/gamma)
    model.fit(x, y)
    print(model.coef_)

def gradient_descent(x, y, gamma):
    m = x.shape[0]
    x = np.insert(x, 0, 1, axis=1)
    paras_init = np.full(x.shape[1], 1)
    updated_paras = paras_init
    alpha = 0.1
    iterator_max_num = 50
    accept_error = 1000

    for iter_num in range(iterator_max_num):
        regularization_paras = updated_paras
        regularization_paras[0] = 1
        descent_list = ((x @ updated_paras - y) @ x + gamma * regularization_paras) / m
        updated_paras = updated_paras - alpha * descent_list
        error = np.sum(np.square(np.dot(x, updated_paras) - y))
        if (error < accept_error):
            break
        print(iter_num, "\t" , error)    
    
    print(updated_paras)

def main():
    x, y, coef = make_regression(n_samples=200, n_features=5, n_informative=50 ,random_state=0, coef=True)
    standard_ridge(x, y, 10)
    gradient_descent(x, y, 1)

if __name__ == '__main__':
    main()