from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from network_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from sklearn.model_selection import train_test_split

def layer_sizes(X, Y):
    n_x = X.shape[0] # size of input layer
    n_h = 4 # size of hidden layer
    n_y = Y.shape[0] # size of output layer
    return (n_x, n_h, n_y)

def my_splitter(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y.T, test_size=0.3, random_state=1)
    X_train, X_test, Y_train, Y_test = X_train.T, X_test.T, Y_train.T, Y_test.T
    return X_train, X_test, Y_train, Y_test

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(0) # we set up a seed so that your output matches ours although the initialization is random.
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def classify_with_lr(X_train, X_test, Y_train, Y_test):
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X_train.T, Y_train.T)

    plot_decision_boundary(lambda x: clf.predict(x), X_test, Y_test)
    plt.title("Logistic Regression")

    # Print accuracy
    LR_predictions = clf.predict(X_test.T)
    print ('Accuracy of logistic regression: %d ' % float((np.dot(Y_test,LR_predictions) + np.dot(1-Y_test,1-LR_predictions))/float(Y_test.size)*100) +
        '% ' + "(percentage of correctly labelled datapoints)")

def plot_sample(X, Y):
    plt.scatter(X[0, :], X[1, :], c=Y.reshape(Y.shape[1]), s=40)
    plt.show()

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
 
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def compute_cost(A2, Y, parameters):
    m = Y.shape[1] # number of example
    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),1-Y)
    cost = -1 * np.sum(logprobs) * (1./m)
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2-Y
    dW2 = 1./m*(np.dot(dZ2,A1.T))
    db2 = (1./m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.dot(W2.T,dZ2) * (1-np.power(A1,2))
    dW1 = (1./m)*np.dot(dZ1,X.T)
    db1 = (1./m)*np.sum(dZ1,axis=1,keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    return predictions

def hl_comparison_plotter(X_train, X_test, Y_train, Y_test):
    plt.figure(figsize=(16, 32))
    hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
    for i, n_h in enumerate(hidden_layer_sizes):
        plt.subplot(5, 2, i+1)
        plt.title('Hidden Layer of size %d' % n_h)
        parameters = nn_model(X_train, Y_train, n_h, num_iterations = 5000)
        plot_decision_boundary(lambda x: predict(parameters, x.T), X_test, Y_test)
        train_predictions = predict(parameters, X_train)
        test_predictions = predict(parameters, X_test)
        train_accuracy = float((np.dot(Y_train,train_predictions.T) + np.dot(1-Y_train,1-train_predictions.T))/float(Y_train.size)*100)
        test_accuracy = float((np.dot(Y_test,test_predictions.T) + np.dot(1-Y_test,1-test_predictions.T))/float(Y_test.size)*100)
        print ("Accuracy for {} hidden units: Train set {:.2f}%,  Test set {:.2f}%".format(n_h, train_accuracy, test_accuracy))
    plt.show()

def accuracy_plotter(X_train, X_test, Y_train, Y_test, max_hu):
    hidden_layer_test = range(1,max_hu+1)
    train_accuracies = []
    test_accuracies = []
    for n_h in hidden_layer_test:
        parameters = nn_model(X_train, Y_train, n_h, num_iterations = 5000)
        train_predictions = predict(parameters, X_train)
        test_predictions = predict(parameters, X_test)
        train_accuracy = float((np.dot(Y_train,train_predictions.T) + np.dot(1-Y_train,1-train_predictions.T))/float(Y_train.size)*100)
        test_accuracy = float((np.dot(Y_test,test_predictions.T) + np.dot(1-Y_test,1-test_predictions.T))/float(Y_test.size)*100)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        if n_h % 10 == 0:
            print ("Accuracies with",n_h,"hidden units calculated")
    print ("The highest train accracy is",np.max(train_accuracies),"% with",np.argmax(train_accuracies)+1,"hidden units")
    print ("The highest test accracy is",np.max(test_accuracies),"% with",np.argmax(test_accuracies)+1,"hidden units") 
    plt.figure(figsize=(8,6))
    plt.plot(hidden_layer_test, train_accuracies, color='r',label='Train accuracy')
    plt.plot(hidden_layer_test, test_accuracies, color='b',label='Test accuracy')
    plt.grid()
    plt.legend(loc='best')


def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def shallow_deeplearning():
    X, Y = load_planar_dataset()
    shape_X = X.shape
    shape_Y = Y.shape
    m = X.shape[1]  # training set size

    X_train, X_test, Y_train, Y_test = my_splitter(X,Y)
    print(X_train.shape)
    # classify_with_lr(X_train, X_test, Y_train, Y_test)
    n_x, n_h, n_y = layer_sizes(X_train, Y_train)
    parameters = nn_model(X_train, Y_train, 4, num_iterations=10000, print_cost=True)
    # plot_decision_boundary(lambda x: predict(parameters, x.T), X_train, Y_train, title="Decision Boundary for hidden layer size " + str(4))
    # plot_decision_boundary(lambda x: predict(parameters, x.T), X_test, Y_test, title="Decision Boundary for hidden layer size " + str(4))
    # hl_comparison_plotter(X_train, X_test, Y_train, Y_test)

def main():
    shallow_deeplearning()


if __name__ == '__main__':
    main()