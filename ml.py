import sys
import math
import numpy as np


class Training:
    def __init__(self, alpha):
        self.alpha = alpha
        self.train_errors = []
        self.valid_errors = []

    @property
    def epoch(self):
        return len(self.train_errors)


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
    if len(datasets) == 0:
        return dataset1[indexes]
    else:
        return [dataset1[indexes]] + [dataset[indexes] for dataset in datasets]


def max_epochs(max_epoch):
    def max_epoch_stopping_criteria(train_data):
        return max_epoch <= train_data.epoch
    return max_epoch_stopping_criteria


def max_error(max_error, min_epoch=None, max_epoch=None):
    def max_error_stopping_criteria(train_data):
        if min_epoch is not None and train_data.epoch - 1 <= min_epoch:
            return False
        if max_epoch is not None and train_data.epoch - 1 > max_epoch:
            return True
        return np.mean(train_data.train_errors[-10:]) < max_error
    return max_error_stopping_criteria


def train_error_change(error_delta, min_epoch=None, max_epoch=None):
    def error_change_stopping_criteria(train_data):
        if min_epoch is not None and train_data.epoch - 1 <= min_epoch:
            return False
        if max_epoch is not None and train_data.epoch - 1 > max_epoch:
            return True
        trn_errs = train_data.train_errors
        error_deltas = [p - n for p, n in zip(trn_errs[:-1], trn_errs[1:])]
        return np.sum(error_deltas[-100:]) < error_delta
    return error_change_stopping_criteria


def cross_validate_data_generator(data, segment_count):
    """Generates `segment_count` segment tuples consisting of training and
    validation data.
    :param data: numpy matrix where row represents sample vectors
    :param segment_count: how many training/validation pairs should be
    generated
    """
    segment_length = int(len(data) / segment_count)
    for start in range(0, len(data), segment_length):
        validation_idx = range(start, start + segment_length)
        training_idx = [index for index in range(len(data))
                        if index < start or index >= start + segment_length]
        yield data[training_idx, :], data[validation_idx, :]


class CrossValidator:
    def __init__(self, algorithm, *init_args, **init_kwargs):
        self.alg = algorithm
        self.args, self.kwargs = init_args, init_kwargs

    def run(self, data, k, *args, **kwargs):
        data = shuffle_data(data)
        best_model = (sys.float_info.max, None)
        for train, validation in cross_validate_data_generator(data, k):
            model = self.create_model()
            error = self.train(model, train, validation, args, kwargs)
            if error < best_model[0]:
                best_model = error, model

        return best_model[1]

    def create_model(self):
        return self.alg(*self.args, **self.kwargs)

    def train(self, model, train, validation,
              model_args=tuple(), model_kwargs={}):
        model.train(train, validation, *model_args, **model_kwargs)
        return model.cost(validation)


class ForwardFeedNetwork:
    def __init__(self, arch, activation='sigmoid'):
        self.arch = arch
        self.activation = activation
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

    def split(self, data):
        if len(data.shape) == 1:
            return data[:self.arch[0]], data[-self.arch[-1]:]
        else:
            return data[:, :self.arch[0]], data[:, -self.arch[-1]:]

    def forward_feed(self, in_, all_activations=False):
        activations = [in_]
        for weight, bias in zip(self.weights, self.biases):
            activ = np.dot(activations[-1], weight) + bias
            activations.append(self.activate(activ))
        return activations if all_activations else activations[-1]

    def cost(self, data):
        input_data, target_data = self.split(data)
        output = self.forward_feed(input_data)
        errors = quadratic_error(output, target_data)
        return np.sum(errors)

    def train(self, data, validation, alpha, stopping_criteria):
        training = Training(alpha)
        while True:
            data = shuffle_data(data)
            for vector in data:
                self.update(vector, alpha)

            training.train_errors.append(self.cost(data))
            if validation is not None:
                training.valid_errors.append(self.cost(validation))

            if stopping_criteria(training):
                break
        return training

    def update(self, vector, alpha):
        sample, target = vector[:self.arch[0]], vector[-self.arch[-1]:]
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
