import ml
import numpy as np
import mnisttk
import logging
from os.path import join


def load_train_data(path):
    def classify(num):
        result = np.zeros(10)
        result[num] = 255.0
        return result

    data = mnisttk.decode(join(path, 'train-images.idx3-ubyte'))
    labels_ = mnisttk.decode(join(path, 'train-labels.idx1-ubyte'))
    labels = np.asarray([classify(n) for n in labels_])
    available_data = np.zeros((60000, 28*28+10), dtype=np.float32)
    available_data[:, :28*28] = data.reshape((60000, 28*28))
    available_data[:, 28*28:] = labels
    ml.normalize(available_data, (0.0, 255.0))
    return available_data


def load_test_data(path):
    def classify(num):
        result = np.zeros(10)
        result[num] = 255.0
        return result

    data = mnisttk.decode(join(path, 't10k-images.idx3-ubyte'))
    labels_ = mnisttk.decode(join(path, 't10k-labels.idx1-ubyte'))
    labels = np.asarray([classify(n) for n in labels_])
    available_data = np.zeros((10000, 28*28+10), dtype=np.float32)
    available_data[:, :28*28] = data.reshape((10000, 28*28))
    available_data[:, 28*28:] = labels
    ml.normalize(available_data, (0.0, 255.0))
    return available_data


logging.basicConfig(level=logging.INFO)


ann = ml.ForwardFeedNetwork((28*28, 100, 10))
data = load_train_data('/home/stderr/.mnist')

ann.train(data, None, 0.001, ml.max_epochs(100))
print(ann.classification_test(load_test_data('/home/stderr/.mnist')))