import keras as keras
import numpy as np

from dataset_manager import load_data

# test model results
def test(path):
    model = keras.models.load_model(path)

    data, labels = load_data()[:2]

    for i in range(10000):
        if np.argmax(labels[i]) < 2:
            continue
        res = model.predict(data[i].reshape(1, 90, -1))
        print(f"Res: {res}", np.argmax(res), np.max(res))
        print(f"Answer: {labels[i]}", np.argmax(labels[i]))
        # print(f"Error: {res - labels[i]}")
        input()


if __name__ == "__main__":
    path = 'models/test.keras'

    test(path)