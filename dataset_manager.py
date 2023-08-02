import numpy as np


# load data for model
def load_data():
    path = r"E:\repos\pycharmRepo\networks\StockPredictor\data\data.npy"  # full path cus can be used in other project
    test_frac = 0.9

    with open(path, 'rb') as f:
        x_data = np.load(f)
        y_data = np.load(f)

    # try to be same amount of liquid and non-liquid examples
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

    length = len(x_data)

    x_train, x_test = np.split(x_data, [int(length * test_frac)])
    y_train, y_test = np.split(y_data, [int(length * test_frac)])

    print(len(x_train))

    return x_train, y_train, x_test, y_test, None, None


if __name__ == "__main__":
    load_data()
