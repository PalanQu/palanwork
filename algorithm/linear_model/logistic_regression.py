import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

def standard_lr(x, y):
    model = LogisticRegression()
    model.fit(x, y)
    print(model.coef_)




def main():
    x, y = make_classification(n_samples=200, n_features=5, random_state=0)
    standard_lr(x, y)

if __name__ == '__main__':
    main()