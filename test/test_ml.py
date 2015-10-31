import ml
import numpy as np
import pytest
from collections import namedtuple


def test_sigmoid_activation_function():
    assert ml.sigmoid(0.0) == 0.5
    assert abs(ml.sigmoid(-10)) < 0.001
    assert abs(ml.sigmoid(10) - 1.0) < 0.001


def test_quadratic_error_scalar():
    assert ml.quadratic_error(1, 0.5) == 0.125


def test_batch_generator():
    array = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    batches = list(ml.batches(array, 4))
    assert len(batches) == 3
    assert np.all(batches[0] == [1, 2, 3, 4])
    assert np.all(batches[1] == [5, 6, 7, 8])
    assert np.all(batches[2] == [9, 10, 11])


def test_quadratic_error_vector():
    output = np.asarray([0.5, 1.0, 0.25])
    target = np.asarray([1.0, 0.1, 0.25])
    expected_errors = [0.125, 0.405, 0.0]
    assert np.all(ml.quadratic_error(output, target) == expected_errors)


def test_quadratic_error_matrix():
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


def test_cross_validate_data_generator():
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


def test_max_epoch_stopping_criteria():
    Trainer = namedtuple('Trainer', ['epoch'])
    stopping_function = ml.max_epoch(10)
    assert stopping_function(Trainer(5)) is False
    assert stopping_function(Trainer(10)) is True
    assert stopping_function(Trainer(15)) is True


def test_mlp_ann_init_weights():
    ann = ml.ForwardFeedNetwork((3, 2, 4), 'sigmoid')
    assert len(ann.weights) == 2
    assert ann.weights[0].shape == (3, 2)
    assert ann.weights[1].shape == (2, 4)


def test_mlp_ann_init_random_weights():
    ann1 = ml.ForwardFeedNetwork((3, 2, 4), 'sigmoid')
    ann2 = ml.ForwardFeedNetwork((3, 2, 4), 'sigmoid')
    for w1, w2 in zip(ann1.weights, ann2.weights):
        assert np.any(w1 != w2)


def test_mlp_ann_init_biases():
    ann = ml.ForwardFeedNetwork((3, 2, 4), 'sigmoid')
    assert len(ann.biases) == 2
    assert ann.biases[0].shape == (2, )
    assert ann.biases[1].shape == (4, )


def test_mlp_ann_init_random_biases():
    ann1 = ml.ForwardFeedNetwork((3, 2, 4), 'sigmoid')
    ann2 = ml.ForwardFeedNetwork((3, 2, 4), 'sigmoid')
    for b1, b2 in zip(ann1.biases, ann2.biases):
        assert np.any(b1 != b2)


def test_mlp_ann_forward_feed_vector():
    ann = ml.ForwardFeedNetwork((3, 2, 4))
    output = ann.forward_feed(np.asarray([1, 2, 3]))
    assert output.shape == (4, )


def test_mlp_ann_forward_feed_vector_all():
    ann = ml.ForwardFeedNetwork((3, 2, 4))
    activations = ann.forward_feed(np.asarray([1, 2, 3]), all_activations=True)
    assert len(activations) == 3
    assert activations[0].shape == (3, )
    assert np.all(activations[0] == [1, 2, 3])
    assert activations[1].shape == (2, )
    assert activations[2].shape == (4, )


def test_mlp_ann_forward_feed_matrix():
    ann = ml.ForwardFeedNetwork((3, 2, 4))
    input_matrix = np.asarray([1, 2,  3, 4, 5, 6]).reshape((2, 3))
    output = ann.forward_feed(input_matrix)
    assert output.shape == (2, 4)


def test_mlp_ann_forward_feed_output_bounds():
    ann = ml.ForwardFeedNetwork((3, 2, 4))
    input_matrix = np.asarray([-100, -50,  10, 10, 50, 100]).reshape((2, 3))
    output = ann.forward_feed(input_matrix)
    assert np.all(output > 0)
    assert np.all(output < 1)


def test_mlp_ann_update_for_sample_sanity():
    ann = ml.ForwardFeedNetwork((2, 2, 1))
    in_, target = np.asarray([1.0, 0.0]), np.asarray([1.0])
    ann.update(in_, target, 0.2)


def test_mlp_ann_train_sanity(xor_dataset):
    ann = ml.ForwardFeedNetwork((2, 2, 1))
    ann.train(xor_dataset[:, :2], xor_dataset[:, 2:], 0.2, ml.max_epoch(5))
    assert ann.epoch == 5


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
        derivative = (plus_cost - minus_cost) / 2 * epsilon
        updated_parameters.append(derivative)
    return np.asarray(updated_parameters) * 100000000


def test_mlp_ann_gradient_estimation_comparison(xor_dataset):
    ann = ml.ForwardFeedNetwork((2, 2, 1))
    original_parameters = extract_parameters(ann)
    sample, target = xor_dataset[3, :2], xor_dataset[3, 2:]

    delta_est = estimate_parameters(ann, sample, target)
    ann.update(sample, target, 1.0)
    backpropagation_parameters = extract_parameters(ann)

    delta_bcp = original_parameters - backpropagation_parameters
    dp = np.abs(delta_bcp - delta_est)
    assert np.all(dp < 0.00001)

