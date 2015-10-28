import math
import numpy as np


@np.vectorize
def sigmoid(t):
    return 1 / (1 + math.e ** -t)


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


def max_epoch(max_epoch):
    def max_epoch_stopping_criteria(trainer):
        return max_epoch <= trainer.epoch
    return max_epoch_stopping_criteria


class MLP:
    """Multilayer perceptron with hidden layers.
    """
    def __init__(self, arch):
        self.arch = arch
        rand = np.random.rand
        self.weights, self.biases = [], []
        for input_dim, neuron_count in zip(arch[:-1], arch[1:]):
            self.weights.append(rand(input_dim, neuron_count) * 0.1 - 0.05)
            self.biases.append(rand(neuron_count) * 0.1 - 0.05)

    def forward_feed(self, data, all=False):
        activations = [data]
        for w, b in zip(self.weights, self.biases):
            activations.append(sigmoid(np.dot(activations[-1], w) + b))
        return activations if all else activations[-1]


class BackpropagationTeacher:
    def __init__(self, network_to_train, alpha):
        self.network = network_to_train
        self.alpha = alpha
        self.train_errors = []
        self._wr = list(reversed(self.network.weights))
        self._br = list(reversed(self.network.biases))

    @property
    def epoch(self):
        return len(self.train_errors)

    def gradient_descent(self, data, stopping_criterion, batch_size=20):
        """Gradient descent learning algorithm "backed" by backpropagation
        that assumes sigmoidal activation function, which is used by default
        in forward_feed method.
        """
        while True:
            np.random.shuffle(data)
            self.train_errors.append(
                np.mean(self.compute_quadratic_error(data))
            )

            for batch in batches(data, batch_size):
                self.teach_batch(batch)

            if stopping_criterion(trainer=self):
                self.train_errors.append(
                    np.mean(self.compute_quadratic_error(data))
                )
                break

    def teach_batch(self, batch):
        """Use backpropagation once to update weights and biases in order to
        decrease error on provided sample.
        """
        net = self.network
        if batch.shape[1] != net.arch[0] + net.arch[-1]:
            raise ValueError('Batch doesn\'t have correct dimensions')
        input_data, target = batch[:, :net.arch[0]], batch[:, net.arch[0]:]

        activations = net.forward_feed(input_data, all=True)
        output, *hidden_output = reversed(activations)

        # Compute error for last layer
        errors = [(output - target) * output * (1 - output)]
        # Compute error for the rest of layers
        for w, activ in zip(self._wr, hidden_output):
            err = np.dot(w, errors[-1].T) * (activ * (1 - activ)).T
            errors.append(err.T)

        update_data = zip(net.weights, net.biases, reversed(errors[:-1]),
                          activations[:-1])
        for weights, biases, err, activ in update_data:
            biases -= np.mean(self.alpha * err, axis=0)
            weights -= self.alpha * np.dot(activ.T, err) / activ.shape[0]

    def compute_quadratic_error(self, data):
        net = self.network
        if data.shape[1] != net.arch[0] + net.arch[-1]:
            raise ValueError('Data doesn\'t have correct dimensions')
        input_data, target = data[:, :net.arch[0]], data[:, net.arch[0]:]
        output = net.forward_feed(input_data)
        return quadratic_error(output, target)
