import numpy as np


# load raw data from file
def load_raw_data(path):
    with open(path, 'rb') as f:
        x_data = np.load(f)
        y_data = np.load(f)
    return x_data, y_data


# try to make same amount of correct and incorrect examples
def balance_data(x_data, y_data):
    needed_ind = np.in1d(y_data[:, 1], [0])
    needed_x = np.array(x_data[needed_ind])
    needed_y = np.array(y_data[needed_ind])

    indices = np.arange(len(x_data))
    np.random.shuffle(indices)
    # random samples = x2 of high liquid
    x_data = np.concatenate([x_data[indices][:len(needed_x) * 2], needed_x])
    y_data = np.concatenate([y_data[indices][:len(needed_x) * 2], needed_y])

    indices = np.arange(len(x_data))
    np.random.shuffle(indices)
    x_data = x_data[indices]
    y_data = y_data[indices]

    return x_data, y_data


# split data: train and test
def split_data(x_data, y_data, train_fraction):
    length = len(x_data)
    x_train, x_test = np.split(x_data, [int(length * train_fraction)])
    y_train, y_test = np.split(y_data, [int(length * train_fraction)])
    return x_train, y_train, x_test, y_test


# load data for model
def load_data():
    path = r"data/data.npy"  # path to saved data
    train_fraction = 0.9  # part of data used for train

    x_data, y_data = load_raw_data(path)
    x_data, y_data = balance_data(x_data, y_data)
    x_train, y_train, x_test, y_test = split_data(x_data, y_data, train_fraction)

    print(len(x_train))

    return x_train, y_train, x_test, y_test, None, None


if __name__ == "__main__":
    load_data()
