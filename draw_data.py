from datetime import datetime

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import utils
from dataset_manager import load_data
from preprocess import read_csv, process_raw


# load data from csv (raw data)
def show_csv(step=30):
    files = utils.get_data_files() # csv files

    before_days = 90 # input days
    after_days = 30 # output days / needed future

    speed = 5 # how many days skip per step
    time_delay = 1 # time between steps
    sc = MinMaxScaler() # scaler
    date_format = "%Y-%m-%d"
    parameters = utils.parameters

    # parameters / normalized / answers
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.autofmt_xdate(rotation=45) # rotate date

    for file in files:
        # check if user closed window
        if not plt.fignum_exists(fig.number):
            break
        fig.suptitle(file.split('_')[2])

        dates = list(map(lambda x: datetime.strptime(x, date_format), read_csv(file, ['Date'])[:, 0]))
        data = read_csv(file, parameters)

        for i in range(before_days, len(data) - after_days, speed):
            if not plt.fignum_exists(fig.number):
                break

            train_dates = dates[i - before_days:i]
            label_dates = dates[i - 1:i + after_days:step]

            raw_train = data[i - before_days:i]
            raw_label = data[i - 1:i + after_days:step, 0]

            train, label = process_raw(sc, raw_train, raw_label)

            # ax1
            ax1.clear()
            ax1.set_title("Data")
            ax1.plot(train_dates,
                     raw_train,
                     label=parameters)
            ax1.axvline(train_dates[-1], color='red')
            ax1.plot(label_dates, raw_label, color='red')
            ax1.legend()

            # ax2
            ax2.clear()
            ax2.set_title("Normalized")
            ax2.plot(train_dates,
                     train,
                     label=parameters)
            ax2.legend()

            # ax3
            ax3.clear()
            ax3.set_title("Answer")

            # bars
            ax3.bar(range(len(label)), label)
            # graph
            # ax3.plot(label_dates, label, label='Close', color='red')

            # show plot
            fig.autofmt_xdate()
            plt.draw()
            plt.pause(time_delay)

# load data from file (preprocessed)
def show_data(step=30):
    train, labels = load_data()[:2]
    train_dates = [i for i in range(1, 91)]
    label_dates = [i for i in range(1, 31, step)]
    # normalized / answers
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    for i in range(len(train)):
        if not plt.fignum_exists(fig.number):
            break

        # ax1
        ax1.clear()
        ax1.set_title("Normalized")
        ax1.plot(train_dates, train[i])

        # ax2
        ax2.clear()
        ax2.set_title("Answer")

        # bars
        ax2.bar(range(len(labels[i])), labels[i])
        # graph
        # ax2.plot(label_dates, labels[i], color='red')

        # show plot
        fig.autofmt_xdate()
        plt.draw()
        plt.pause(1)

# show data
def draw(use_csv=True, step=1):
    if use_csv:
        show_csv(step=step)
    else:
        show_data(step=step)


if __name__ == "__main__":
    draw(use_csv=True, step=utils.step)
