import pytest
from os.path import abspath, dirname, join


@pytest.fixture(scope='module')
def resources_root():
    return join(abspath(dirname(__file__)), 'resources')
