import os
import numpy as np
import string
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from neural_network import NeuralNetwork
from converter import Converter

TARGET_NAMES = list(string.ascii_uppercase)


def to_vector(y, l):
    arr = np.zeros([y.shape[0], l])
    for i in range(0, y.shape[0]):
        arr[i, int(y[i])] = 1
    return arr


def prepare_data(path, size, count_of_examples, count_of_classes):
    x = np.zeros((count_of_examples, size))
    y = np.zeros((count_of_examples, ))
    c = Converter()
    i = -1
    for cls, f0 in enumerate(os.listdir(path)):
        for f1 in os.listdir(os.path.join(path, f0)):
            img = os.path.join(path, f0, f1)
            i += 1
            x[i] = c.img_to_range(img, size, -2, 2)
            y[i] = cls
    return x, to_vector(y, count_of_classes)


def custom_dataset():
    x, y = prepare_data("./Letters", 400, 7800, 26)
    nn = NeuralNetwork([400, int(400*0.8), 26])
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    nn.back_propagation(X_train, y_train, epochs=10, learning_rate=0.1)
    y_predicted = nn.predict(X_test)
    y_predicted = np.argmax(y_predicted, axis=1).astype(int)
    y_test = np.argmax(y_test, axis=1).astype(int)
    print(metrics.classification_report(y_test, y_predicted, target_names=TARGET_NAMES))


def mnist_dataset():
    digits = datasets.load_digits()
    X = preprocessing.scale(digits.data.astype(float))
    y = to_vector(digits.target, 10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    nn = NeuralNetwork([64, 60, 10])
    nn.back_propagation(X_train, y_train, epochs=15, learning_rate=0.1)
    y_predicted = nn.predict(X_test)
    y_predicted = np.argmax(y_predicted, axis=1).astype(int)
    y_test = np.argmax(y_test, axis=1).astype(int)
    print(metrics.classification_report(y_test, y_predicted))


if __name__ == '__main__':
    mnist_dataset()
    custom_dataset()
