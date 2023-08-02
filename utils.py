import os

raw_data_folder = "raw_data"
data_file = "data/data.npy"
tickers_list = "tickers"

step = 30  # step between days in output of model

parameters = {
    'Close': True,
    'High': True,
    'Low': True,
    'Open': False,
    'Adj Close': False,
    'Volume': False
}  # which columns needed of csv files
parameters = [key for key in parameters if parameters[key]]


# get tickers from file
def get_tickers():
    tickers = []
    with open(tickers_list, 'r') as f:
        for line in f:
            tickers.append(line.split()[0])

    return tickers


# get list of csv files in raw_data folder
def get_data_files():
    return os.listdir("raw_data")


if __name__ == "__main__":
    print(get_tickers())
