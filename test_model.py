import keras as keras
import numpy as np

from dataset_manager import load_data

# test model results
def test(path):
    model = keras.models.load_model(path)

    data, labels = load_data()[:2]

    for i in range(10000):
        res = model.predict(data[i].reshape(1, 90, -1))
        model_ans = np.argmax(res)
        assurance = np.max(res)

        print(f"Model output: {res}, Model answer: {model_ans}, Assurance: {assurance:.4f}")
        print(f"Correct output: {labels[i]}, Correct answer: {np.argmax(labels[i])}")
        print(f"Errors: {res - labels[i]}")
        input("Press enter to continue")


if __name__ == "__main__":
    path = 'models/test.keras'

    test(path)