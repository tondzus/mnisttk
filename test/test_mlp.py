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


def test_max_epoch_stopping_criteria():
    Trainer = namedtuple('Trainer', ['epoch'])
    stopping_function = ml.max_epoch(10)
    assert stopping_function(Trainer(5)) is False
    assert stopping_function(Trainer(10)) is True
    assert stopping_function(Trainer(15)) is True


def test_mlp_ann_init_weights():
    ann = ml.MLP((3, 2, 4))
    assert len(ann.weights) == 2
    assert ann.weights[0].shape == (3, 2)
    assert ann.weights[1].shape == (2, 4)


def test_mlp_ann_init_random_weights():
    ann1 = ml.MLP((3, 2, 4))
    ann2 = ml.MLP((3, 2, 4))
    for w1, w2 in zip(ann1.weights, ann2.weights):
        assert np.any(w1 != w2)


def test_mlp_ann_init_biases():
    ann = ml.MLP((3, 2, 4))
    assert len(ann.biases) == 2
    assert ann.biases[0].shape == (2, )
    assert ann.biases[1].shape == (4, )


def test_mlp_ann_init_random_biases():
    ann1 = ml.MLP((3, 2, 4))
    ann2 = ml.MLP((3, 2, 4))
    for b1, b2 in zip(ann1.biases, ann2.biases):
        assert np.any(b1 != b2)


def test_mlp_ann_forward_feed_vector():
    ann = ml.MLP((3, 2, 4))
    output = ann.forward_feed(np.asarray([1, 2, 3]))
    assert output.shape == (4, )


def test_mlp_ann_forward_feed_vector_all():
    ann = ml.MLP((3, 2, 4))
    activations = ann.forward_feed(np.asarray([1, 2, 3]), all=True)
    assert len(activations) == 3
    assert activations[0].shape == (3, )
    assert np.all(activations[0] == [1, 2, 3])
    assert activations[1].shape == (2, )
    assert activations[2].shape == (4, )


def test_mlp_ann_forward_feed_matrix():
    ann = ml.MLP((3, 2, 4))
    input_matrix = np.asarray([1, 2,  3, 4, 5, 6]).reshape((2, 3))
    output = ann.forward_feed(input_matrix)
    assert output.shape == (2, 4)


def test_mlp_ann_forward_feed_output_bounds():
    ann = ml.MLP((3, 2, 4))
    input_matrix = np.asarray([-100, -50,  10, 10, 50, 100]).reshape((2, 3))
    output = ann.forward_feed(input_matrix)
    assert np.all(output > 0)
    assert np.all(output < 1)


@pytest.mark.parametrize("ann", [
    ml.MLP((2, 1, 1)),
    ml.MLP((2, 1, 1)),
    ml.MLP((2, 1, 1)),
    ml.MLP((2, 1, 1)),
    ml.MLP((2, 1, 1)),
    ml.MLP((2, 1, 1)),
    ml.MLP((2, 1, 1)),
    ml.MLP((2, 1, 1)),
    ml.MLP((2, 1, 1)),
    ml.MLP((2, 1, 1)),
])
def test_mlp_ann_gradient_descent_error_reduction(ann, xor_dataset):
    trainer = ml.BackpropagationTeacher(ann, alpha=0.01)
    trainer.gradient_descent(xor_dataset, ml.max_epoch(5))
    assert trainer.train_errors[-1] < trainer.train_errors[0]
