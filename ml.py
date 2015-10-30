import math
import numpy as np
import random


@np.vectorize
def sigmoid(t):
    return 1 / (1 + math.e ** -t)


@np.vectorize
def sigmoid_prime(t):
    return t * (1 - t)


@np.vectorize
def quadratic_error(output, target):
    return (target - output) ** 2 / 2


def batches(data, batch_size):
    """Generates - it really is a generator - batches of size `batch_size` from
    `data`. Data needs to be of type numpy.ndarray or have the same slicing
    interface.
    """
    for start in range(0, len(data), batch_size):
        yield data[start:start+batch_size]


def normalize(data, minmax=None):
    if minmax is None:
        minmax = np.min(data), np.max(data)
    data -= minmax[0]
    data /= minmax[1] - minmax[0]
    return minmax, data


def shuffle_data(dataset1, *datasets):
    count = len(dataset1)
    indexes = np.random.permutation(range(count))
    return [dataset1[indexes]] + [dataset[indexes] for dataset in datasets]


def max_epoch(max_epoch):
    def max_epoch_stopping_criteria(train_data):
        return max_epoch <= train_data.epoch
    return max_epoch_stopping_criteria


def cross_validate_data_generator(data, segment_count):
    """Generates `segment_count` segment tuples consisting of training and
    validation data.
    """
    segment_length = int(len(data) / segment_count)
    for start in range(0, len(data), segment_length):
        validation_idx = range(start, start + segment_length)
        training_idx = [index for index in range(len(data))
                        if index < start or index >= start + segment_length]
        yield data[training_idx, :], data[validation_idx, :]


class ForwardFeedNetwork:
    def __init__(self, arch, activation='sigmoid'):
        self.arch = arch
        if activation == 'sigmoid':
            self.activate = sigmoid
            self.activate_prime = sigmoid_prime
        else:
            raise ValueError('Unsupported activation function')

        rand = np.random.rand
        self.weights, self.biases = [], []
        for in_dim, n_count in zip(arch[:-1], arch[1:]):
            self.weights.append(rand(in_dim, n_count) * 0.1 - 0.05)
            self.biases.append(rand(n_count) * 0.1 - 0.05)

    def forward_feed(self, in_, all_activations=False):
        activations = [in_]
        for weight, bias in zip(self.weights, self.biases):
            activ = np.dot(activations[-1], weight) + bias
            activations.append(self.activate(activ))
        return activations if all_activations else activations[-1]

    def classify(self, sample, treshold=0.5):
        activations = self.forward_feed(sample)
        return np.array([0.0 if a < treshold else 1.0
                         for a in activations])

    def train(self, input_data, target_data, alpha, stopping_criteria):
        self.epoch = 0
        while True:
            self.epoch += 1
            input_data, target_data = shuffle_data(input_data, target_data)
            for in_, target in zip(input_data, target_data):
                self.update(in_, target, alpha)

            if stopping_criteria(self):
                break

    def update(self, sample, target, alpha):
        activations = self.forward_feed(sample, all_activations=True)
        output, other_activ = activations[-1], reversed(activations[1:-1])
        deltas = [(target - output) * self.activate_prime(output)]
        for a, w in zip(other_activ, reversed(self.weights[1:])):
            delta = np.dot(deltas[-1], w.T) * self.activate_prime(a)
            deltas.append(delta)
        deltas.reverse()

        for w, b, a, d in zip(self.weights, self.biases, activations, deltas):
            b += alpha * d
            w += alpha * (np.matrix(a).T * np.matrix(d))
