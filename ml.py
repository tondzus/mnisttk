import math
import numpy as np
import logging
from multiprocessing import Pool


class Training:
    def __init__(self, alpha):
        self.alpha = alpha
        self.train_errors = []
        self.valid_errors = []

    @property
    def epoch(self):
        return len(self.train_errors)

    def __eq__(self, other):
        return self.train_errors == other.train_errors

    def __hash__(self):
        return hash(self.train_errors)


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
        if min_epoch is not None and min_epoch > train_data.epoch:
            return False
        if max_epoch is not None and max_epoch <= train_data.epoch:
            return True
        return np.mean(train_data.train_errors[-10:]) < max_error
    return max_error_stopping_criteria


def train_error_change(error_delta, min_epoch=None, max_epoch=None):
    def error_change_stopping_criteria(train_data):
        if min_epoch is not None and min_epoch > train_data.epoch:
            return False
        if max_epoch is not None and max_epoch <= train_data.epoch:
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
        self.model_args, self.model_kwargs = tuple(), dict()

    def run(self, data, k, *args, **kwargs):
        self.model_args, self.model_kwargs = args, kwargs
        data, models = shuffle_data(data), []
        for train_data in cross_validate_data_generator(data, k):
            result = self.train_model_for(train_data)
            models.append(result)
        return models

    def create_model(self):
        return self.alg(*self.args, **self.kwargs)

    def train(self, model, train, validation):
        model.train(train, validation, *self.model_args, **self.model_kwargs)
        return model.cost(validation)

    def train_model_for(self, data):
        train_data, validation_data = data
        log = logging.getLogger(__name__)
        model = self.create_model()
        error = self.train(model, train_data, validation_data)
        log.info('Finished training model')
        return error, model


class ForwardFeedNetwork:
    def __init__(self, arch, activation='sigmoid'):
        self.arch = arch
        self.activation = activation
        self._set_activation()
        self.last_training = None
        rand = np.random.rand
        self.weights, self.biases = [], []
        for in_dim, n_count in zip(arch[:-1], arch[1:]):
            self.weights.append(rand(in_dim, n_count) * 0.1 - 0.05)
            self.biases.append(rand(n_count) * 0.1 - 0.05)

    def _set_activation(self):
        if self.activation == 'sigmoid':
            self.activate = sigmoid
            self.activate_prime = sigmoid_prime
        else:
            raise ValueError('Unsupported activation function')

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
        # return np.sum(np.mean(errors, axis=0))
        return np.mean(errors)

    def train(self, data, validation, alpha, stopping_criteria):
        log = logging.getLogger(__name__)
        log.info('Training started!')
        self.last_training = Training(alpha)
        while True:
            data = shuffle_data(data)
            for vector in data:
                self.update(vector, alpha)

            trn_err = self.cost(data)
            self.last_training.train_errors.append(trn_err)
            msg = '{}. epoch: training - {}'.format(self.last_training.epoch, trn_err)
            if validation is not None:
                vld_err = self.cost(validation)
                self.last_training.valid_errors.append(vld_err)
                msg += ', validation - {}'.format(vld_err)
            log.info(msg)

            if stopping_criteria(self.last_training):
                break
        return self.last_training

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

    def classification_test(self, data):
        outputs = np.argmax(self.forward_feed(data[:, :self.arch[0]]), axis=1)
        targets = np.argmax(data[:, -self.arch[-1]:], axis=1)
        return np.sum(targets - outputs == 0), len(outputs)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['activate']
        del state['activate_prime']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._set_activation()
