import ml
import numpy as np
import pytest
import mnisttk
from os.path import abspath, dirname, join


@pytest.fixture(scope='module')
def resources_root():
    return join(abspath(dirname(__file__)), 'resources')


@pytest.fixture(scope='function')
def and_dataset(resources_root):
    and_dataset_path = join(resources_root, 'and_dataset.idx')
    return mnisttk.decode(and_dataset_path)


@pytest.fixture(scope='function')
def xor_dataset(resources_root):
    xor_dataset_path = join(resources_root, 'xor_dataset.idx')
    return mnisttk.decode(xor_dataset_path)


@pytest.fixture(scope='function')
def rigged_ann():
    ann = ml.MLP((2, 2, 1))
    ann.weights = [
        np.asarray([[0.1, -0.1],
                    [-0.1, 0.1]]),
        np.asarray([[0.1],
                    [-0.1]]),
    ]
    ann.biases = [
        np.asarray([-0.3, 0.3]),
        np.asarray([0.1]),
    ]
    return ann
