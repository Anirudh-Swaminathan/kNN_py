""" This is the data generator file
It generates random data which conforms to different classes
for classification
"""
import numpy as np

def generate_data(num_classes, num_samps):
    """ Function to generate data randomly

    Keyword arguments:
    num_classes -- the number of classes
    num_samps -- the number of samples in each class
    """

    # The number of test samples in each class
    te_samps = num_samps/5

    # The total number of random outliers in the data
    ran_samps = num_samps/10
    train_X = np.zeros(shape=(num_classes*num_samps,2))
    test_X = np.zeros(shape=(num_classes*te_samps, 2))
    train_y = np.array([None for _ in range(num_classes*num_samps)])
    # train_y = np.zeros(shape=(num_classes*num_samps,))
    test_y = np.array([None for _ in range(num_classes*te_samps)])
    # test_y = np.zeros(shape=(num_classes*te_samps,))

    # The range in which each class of data lies in
    range_val = 23
    for i in range(num_classes):
        start_val = i*range_val
        end_val = start_val + range_val - 3
        # print start_val, end_val
        train_X[i*num_samps:(i+1)*num_samps, :] = np.random.uniform(
                low=start_val, high=end_val, size=(num_samps, 2))
        train_y[i*num_samps:(i+1)*num_samps] = np.array(
                [chr(ord('A') + i) for _ in range(num_samps)])
        test_X[i*te_samps:(i+1)*te_samps, :] = np.random.uniform(
                low=start_val, high=end_val, size=(te_samps, 2))
        test_y[i*te_samps:(i+1)*te_samps] = np.array(
                [chr(ord('A') + i) for _ in range(te_samps)])
    # print train_X.shape, test_X.shape, train_y.shape, test_y.shape
    train_data = np.column_stack((train_X, train_y))
    test_data = np.column_stack((test_X, test_y))

    # Generate the random outliers
    if ran_samps:
        rand_X = np.random.uniform(
                low=0.0, high=(num_classes-1)*range_val, size=(ran_samps, 2))
        rand_y = np.array(
                [chr(ord('A') + np.random.randint(low=0, high=num_classes))
                for _ in range(ran_samps)])
        rand_data = np.column_stack((rand_X, rand_y))
        train_data = np.row_stack(tup=(train_data, rand_data))
    np.savetxt(fname='train.csv', X=train_data, delimiter=',')
    np.savetxt(fname='test.csv', X=test_data, delimiter=',')
    # print train_data
    # print test_data

def main():
    num_classes = int(raw_input("Enter the number of classes to generate: "))
    if num_classes < 2 or num_classes > 10:
        num_classes = 3
    num_samps = int(raw_input("Enter the number of samples in each classes: "))
    if num_samps < 25 or num_samps > 750:
        num_samps = 50
    generate_data(num_classes, num_samps)

if __name__ == '__main__':
    main()
