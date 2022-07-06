from pip import main


import numpy as np
from numpy_utils import save_array
from keras.datasets import mnist
from keras.utils import np_utils

MNIST_PATH = "../datasets/mnist"


def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]


def main():
    # load MNIST from server
    print("Loading MNIST from server...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("Done.")
    # preprocess data
    print("Preprocessing data...")
    x_train, y_train = preprocess_data(x_train, y_train, 60_000)
    x_test, y_test = preprocess_data(x_test, y_test, 10_000)
    print("Done.")
    # save data to files
    print("Saving data to files...")
    save_array(x_train, MNIST_PATH + "/x_train.tzn")
    save_array(y_train, MNIST_PATH + "/y_train.tzn")
    save_array(x_test, MNIST_PATH + "/x_test.tzn")
    save_array(y_test, MNIST_PATH + "/y_test.tzn")
    print("Done.")

if(__name__ == "__main__"):
    main()
