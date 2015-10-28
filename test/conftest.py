import pytest
import mnisttk
from os.path import abspath, dirname, join


@pytest.fixture(scope='module')
def resources_root():
    return join(abspath(dirname(__file__)), 'resources')


@pytest.fixture(scope='module')
def xor_dataset(resources_root):
    xor_dataset_path = join(resources_root, 'xor_dataset.idx')
    return mnisttk.decode(xor_dataset_path)
