import time

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

import utils

# download ticker s and save to csv
def download():
    tickers = utils.get_tickers()
    start_date = '2010-01-01'
    end_date = '2020-01-01'

    files = utils.get_data_files()

    for ticker in tickers:
        file_name = f"raw_data_{ticker}_{start_date}_{end_date}.csv"

        if file_name not in files:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            stock_data.to_csv(f"{utils.raw_data_folder}/{file_name}")

# create samples from csv
def make_batches():
    files = utils.get_data_files()

    before_days = 90
    after_days = 30
    sc = MinMaxScaler()

    X = []
    Y = []

    for file in files:
        data = read_csv(file, utils.parameters)
        for i in range(before_days, len(data) - after_days):
            raw_train = data[i - before_days:i]
            raw_label = data[i:i + after_days:utils.step, 0]
            train, label = process_raw(sc, raw_train, raw_label)

            X.append(train)
            Y.append(label)

    X = np.array(X)
    Y = np.array(Y)

    with open(utils.data_file, 'wb') as f:
        np.save(f, X)
        np.save(f, Y)

# make labels categories
def categorize_percentage_change(percentage_change):
    # tries =(
    # categories = []
    # for i in range(-100, 100):
    #     categories.append([i / 100, (i + 1) / 100])
    # categories = [
    #     (-100, -0.2),
    #     (-0.2, -0.1),
    #     (-0.1, -0.05),
    #     (-0.05, -0.02),
    #     (-0.02, 0),
    #     (0, 0.02),
    #     (0.02, 0.05),
    #     (0.05, 0.1),
    #     (0.1, 0.2),
    #     (0.2, 100),
    # ]
    categories = [
        (-100, -0.07),
        (-0.07, 0.07),
        (0.07, 100),
    ]
    categorized_vector = [0] * len(categories)

    for i, (lower_bound, upper_bound) in enumerate(categories):
        if lower_bound <= percentage_change < upper_bound:
            categorized_vector[i] = 1
            break

    return categorized_vector

# read columns from csv
def read_csv(file, columns):
    return pd.read_csv(f"{utils.raw_data_folder}/{file}")[
        columns
    ].to_numpy()

# process raw data and return x, y
def process_raw(sc, raw_train, raw_label, indx=0):
    last_day_close = raw_train[-1][indx]
    # train = sc.fit_transform(raw_train)
    _min = np.min(raw_train)
    _max = np.max(raw_train)
    train = (raw_train - _min) / (_max - _min)
    # label = raw_label / last_day_close
    label = (raw_label - last_day_close) / last_day_close
    label = categorize_percentage_change(label[-1])
    return train, label


def test():
    print(process_raw(None, np.array([[1, 2]]), np.array([-0.1])))


if __name__ == "__main__":
    t = time.time()
    make_batches()
    print(f"Finished in {time.time() - t}")
