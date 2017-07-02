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

def main():
    train_name = raw_input("Please enter the name of the file containing the \
            training data\n")
    test_name = raw_input("Please enter the name of the file containing the \
            testing data\n")
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

if __name__ == '__main__':
    main()
