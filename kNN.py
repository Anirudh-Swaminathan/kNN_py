""" The program to classify the randomly generated data using the k-Nearest
        Neighbours algorithm
"""
import sys
import numpy as np
import math

def loadData(fname='train.txt'):
    """ Function to load the randomly generated data

    Keyword arguments:
    fname -- The file name to load the data in to the numpy array from (default "train.txt")
    """
    try:
        data = np.loadtxt(fname, delimiter=',')
    except Exception as e:
        sys.exit("Could not open file!!")
    np.random.shuffle(data)
    num_cols = data.shape[1]
    y = data[:, num_cols-1]
    X = np.ones(shape=(len(y),num_cols-1))
    for i in range(num_cols-1):
        X[:,i] = data[:,i]
    X = X.astype(float)
    return X, y

def euclidean_distance(x1, x2):
    """ Function to calculate the Euclidean distance between 2 data points

    Keyword arguments:
    x1 -- the first data point
    x2 -- the second data point
    """
    distance = 0
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)

    return math.sqrt(distance)

def getClass(kNN, classes):
    """ Function to obtain the class of the test data point

    Keyword arguments:
    kNN -- the k nearest neighbours
    classes -- the possible classes
    """
    max_count = 0
    pred_class = None
    for c in np.unique(classes):
        # For each class, count the number of occurances in the nearest neighbours
        # of the data points
        co = len(kNN[kNN[:, 1] == c])
        if co > max_count:
            max_count = co
            pred_class = c
    return pred_class

def predict(X_test, X_train, y_train, k):
    """ Function to predict the classes of the test data

    Keyword arguments:
    X_test -- the test data
    X_train -- the training data
    y_train -- the labels of the training data
    """
    classes = np.unique(y_train)
    predictions = []
    for tex in X_test:
        # For each test sample
        neighbours = []
        for j, trx in enumerate(X_train):
            # For each training point, calculate the eculidean distance of the
            # current test sample from the current training sample, and store
            # the distances as well as the class of that sample
            dist = euclidean_distance(x1=tex, x2=trx)
            y = y_train[j]
            neighbours.append([dist, y])
        neighbours = np.array(neighbours)

        # Sort the list based on ascending order of the distances
        kNN = neighbours[neighbours[:, 0].argsort()][:k]
        pred = getClass(kNN=kNN, classes=classes)
        # print pred
        predictions.append(pred)
    return np.array(predictions)

def accuracy(y, y_pred):
    """ Function to obtain the accuracy of the training and the testing data

    Keyword arguments:
    y -- the original labels of the data
    y_pred -- the predicted labels of the data
    """
    accuracy = np.sum(y == y_pred) / len(y)
    return accuracy

def main():
    train_name = raw_input("Please enter the name of the file containing the "\
            + "training data\n")
    test_name = raw_input("Please enter the name of the file containing the "\
            + "testing data\n")
    train_X, train_y = loadData(fname=train_name)
    test_X, test_y = loadData(fname=test_name)
    try:
        k = int(raw_input("Please enter k, the number of neighbours to compare to: "))
    except Exception as e:
        k = 3
    if k<1 or k>9:
        k = 3
    num_classes = len(np.unique(train_y))
    if k<=num_classes:
        k = num_classes+1
    if not k%2:
        k+=1
    predictions = predict(X_test=test_X, X_train=train_X, y_train=train_y, k=k)
    acc = accuracy(y=test_y, y_pred=predictions)
    print "For k = %d, the testing accuracy is %.5f%%" % (k, acc*100.0)

if __name__ == '__main__':
    main()
