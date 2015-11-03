import ml
import numpy as np
from collections import namedtuple


def update_ann(ann, parameters):
    ann = ml.ForwardFeedNetwork(ann.arch, ann.activation)
    ann.weights, ann.biases = [], []
    start = 0
    for input_dim, neuron_count in zip(ann.arch[:-1], ann.arch[1:]):
        w_count, b_count = input_dim * neuron_count, neuron_count
        weights = np.asarray(parameters[start:start+w_count])
        start += w_count
        biases = np.asarray(parameters[start:start+b_count])
        start += b_count
        ann.weights.append(weights.reshape((input_dim, neuron_count)))
        ann.biases.append(biases)
    return ann


def extract_parameters(ann):
    parameters = []
    for weights, biases in zip(ann.weights, ann.biases):
        parameters.extend(weights.reshape(-1))
        parameters.extend(biases.reshape(-1))
    return np.asarray(parameters)


def estimate_parameters(ann, sample, target, epsilon=0.0001):
    original_parameters = extract_parameters(ann)
    ann = ml.ForwardFeedNetwork(ann.arch, ann.activation)

    updated_parameters = []
    for index in range(len(original_parameters)):
        dp = np.zeros(len(original_parameters))
        dp[index] += epsilon
        minus_p = original_parameters - dp
        ann = update_ann(ann, minus_p)
        minus_errors = ml.quadratic_error(ann.forward_feed(sample), target)
        minus_cost = np.mean(minus_errors)
        plus_p = original_parameters + dp
        ann = update_ann(ann, plus_p)
        plus_errors = ml.quadratic_error(ann.forward_feed(sample), target)
        plus_cost = np.mean(plus_errors)
        derivative = (plus_cost - minus_cost) / (2 * epsilon)
        updated_parameters.append(derivative)
    return np.asarray(updated_parameters)


class TestMathFunctions:
    def test_sigmoid_activation_function(self):
        assert ml.sigmoid(0.0) == 0.5
        assert abs(ml.sigmoid(-10)) < 0.001
        assert abs(ml.sigmoid(10) - 1.0) < 0.001

    def test_quadratic_error_scalar(self):
        assert ml.quadratic_error(1, 0.5) == 0.125

    def test_batch_generator(self):
        array = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        batches = list(ml.batches(array, 4))
        assert len(batches) == 3
        assert np.all(batches[0] == [1, 2, 3, 4])
        assert np.all(batches[1] == [5, 6, 7, 8])
        assert np.all(batches[2] == [9, 10, 11])

    def test_quadratic_error_vector(self):
        output = np.asarray([0.5, 1.0, 0.25])
        target = np.asarray([1.0, 0.1, 0.25])
        expected_errors = [0.125, 0.405, 0.0]
        assert np.all(ml.quadratic_error(output, target) == expected_errors)

    def test_quadratic_error_matrix(self):
        output = np.asarray([
            [1.0, 0.5, 0.25],
            [0.5, 1.0, 0.25],
        ])
        target = np.asarray([
            [0.1, 1.0, 0.25],
            [1.0, 0.1, 0.25],
        ])
        expected_errors = [[0.405, 0.125, 0.0], [0.125, 0.405, 0.0]]
        assert np.all(ml.quadratic_error(output, target) == expected_errors)


class TestCrossValidation:
    def test_cross_validate_data_generator(self):
        data = np.reshape(range(12), (6, 2))
        validation_data = list(ml.cross_validate_data_generator(data, 3))
        assert len(validation_data) == 3
        first, second, third = validation_data
        assert np.all(first[0] == [[4, 5], [6, 7], [8, 9], [10, 11]])
        assert np.all(first[1] == [[0, 1], [2, 3]])
        assert np.all(second[0] == [[0, 1], [2, 3], [8, 9], [10, 11]])
        assert np.all(second[1] == [[4, 5], [6, 7]])
        assert np.all(third[0] == [[0, 1], [2, 3], [4, 5], [6, 7]])
        assert np.all(third[1] == [[8, 9], [10, 11]])


class TestStoppingCriteria:
    def test_max_epoch_stopping_criteria(self):
        Trainer = namedtuple('Trainer', ['epoch'])
        stopping_function = ml.max_epochs(10)
        assert not stopping_function(Trainer(9))
        assert stopping_function(Trainer(10))
        assert stopping_function(Trainer(11))

    def test_max_error_stopping_criteria(self):
        Trainer = namedtuple('Trainer', ['epoch', 'train_errors'])
        errors_fail = [0.5, 0.4, 0.3, 0.2, 0.1]
        errors_pass = [0.1, 0.1, 0.1, 0.1, 0.05]
        stopping_function = ml.max_error(0.1, max_epoch=10)
        assert not stopping_function(Trainer(5, errors_fail))
        assert stopping_function(Trainer(5, errors_pass))

    def test_error_change_stopping_criteria(self):
        Trainer = namedtuple('Trainer', ['epoch', 'train_errors'])
        errors_fail = [0.5, 0.4, 0.3, 0.2, 0.1]
        errors_pass = [0.1, 0.1, 0.1, 0.1, 0.09]
        stopping_function = ml.train_error_change(0.02, min_epoch=10)
        assert not stopping_function(Trainer(4, []))
        assert not stopping_function(Trainer(15, errors_fail))
        assert stopping_function(Trainer(15, errors_pass))


class TestForwardFeedNetwork:
    def test_mlp_ann_init_weights(self):
        ann = ml.ForwardFeedNetwork((3, 2, 4), 'sigmoid')
        assert len(ann.weights) == 2
        assert ann.weights[0].shape == (3, 2)
        assert ann.weights[1].shape == (2, 4)

    def test_mlp_ann_init_random_weights(self):
        ann1 = ml.ForwardFeedNetwork((3, 2, 4), 'sigmoid')
        ann2 = ml.ForwardFeedNetwork((3, 2, 4), 'sigmoid')
        for w1, w2 in zip(ann1.weights, ann2.weights):
            assert np.any(w1 != w2)

    def test_mlp_ann_init_biases(self):
        ann = ml.ForwardFeedNetwork((3, 2, 4), 'sigmoid')
        assert len(ann.biases) == 2
        assert ann.biases[0].shape == (2, )
        assert ann.biases[1].shape == (4, )

    def test_mlp_ann_init_random_biases(self):
        ann1 = ml.ForwardFeedNetwork((3, 2, 4), 'sigmoid')
        ann2 = ml.ForwardFeedNetwork((3, 2, 4), 'sigmoid')
        for b1, b2 in zip(ann1.biases, ann2.biases):
            assert np.any(b1 != b2)

    def test_mlp_ann_split_vector(self):
        ann = ml.ForwardFeedNetwork((2, 1))
        vector = np.asarray([0.0, 1.0, 2.0])
        split = ann.split(vector)
        assert np.all(split[0] == [0.0, 1.0])
        assert np.all(split[1] == [2.0])

    def test_mlp_ann_split_matrix(self):
        ann = ml.ForwardFeedNetwork((2, 1))
        matrix = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        split = ann.split(matrix)
        assert np.all(split[0] == [[0.0, 1.0], [3.0, 4.0]])
        assert np.all(split[1] == [[2.0], [5.0]])

    def test_mlp_ann_forward_feed_vector(self):
        ann = ml.ForwardFeedNetwork((3, 2, 4))
        output = ann.forward_feed(np.asarray([1, 2, 3]))
        assert output.shape == (4, )

    def test_mlp_ann_forward_feed_vector_all(self):
        ann = ml.ForwardFeedNetwork((3, 2, 4))
        activations = ann.forward_feed(np.asarray([1, 2, 3]), all_activations=True)
        assert len(activations) == 3
        assert activations[0].shape == (3, )
        assert np.all(activations[0] == [1, 2, 3])
        assert activations[1].shape == (2, )
        assert activations[2].shape == (4, )

    def test_mlp_ann_forward_feed_matrix(self):
        ann = ml.ForwardFeedNetwork((3, 2, 4))
        input_matrix = np.asarray([1, 2,  3, 4, 5, 6]).reshape((2, 3))
        output = ann.forward_feed(input_matrix)
        assert output.shape == (2, 4)

    def test_mlp_ann_forward_feed_output_bounds(self):
        ann = ml.ForwardFeedNetwork((3, 2, 4))
        input_matrix = np.asarray([-100, -50,  10, 10, 50, 100]).reshape((2, 3))
        output = ann.forward_feed(input_matrix)
        assert np.all(output > 0)
        assert np.all(output < 1)

    def test_mlp_ann_update_for_sample_sanity(self):
        ann = ml.ForwardFeedNetwork((2, 2, 1))
        vector = np.asarray([1.0, 0.0, 1.0])
        ann.update(vector, 0.2)

    def test_mlp_ann_train_sanity(self, xor_dataset):
        ann = ml.ForwardFeedNetwork((2, 2, 1))
        training = ann.train(xor_dataset, 0.2, ml.max_epochs(5))
        assert training.epoch == 5

    def test_mlp_ann_gradient_estimation_comparison(self, xor_dataset):
        ann = ml.ForwardFeedNetwork((2, 2, 1))
        original_parameters = extract_parameters(ann)
        sample, target = xor_dataset[3, :2], xor_dataset[3, 2:]

        delta_est = estimate_parameters(ann, sample, target)
        ann.update(xor_dataset[3], 1.0)
        backpropagation_parameters = extract_parameters(ann)

        delta_bcp = original_parameters - backpropagation_parameters
        dp = np.abs(delta_bcp - delta_est)
        assert np.all(dp < 0.00001)

