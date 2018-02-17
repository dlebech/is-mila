import os
import shutil

import pytest

from mila import util


@pytest.fixture
def imagedata(mocker):
    config_mock = mocker.patch('mila.util.config')
    config_mock.IMAGE_DIRECTORY = './tests/images'
    yield config_mock
    shutil.rmtree(config_mock.IMAGE_DIRECTORY)


def test_find_categories(imagedata):
    """It should find all the categories in alphabetical order."""
    for category in ['cat', 'dog']:
        base_dir = os.path.join(imagedata.IMAGE_DIRECTORY, 'all', category)
        os.makedirs(base_dir, exist_ok=True)
    assert util.find_categories() == ['cat', 'dog']


def test_find_categories_different_dir(imagedata):
    """It should find all the categories in a different dir."""
    for category in ['cat', 'dog']:
        base_dir = os.path.join(imagedata.IMAGE_DIRECTORY, 'differentdir', category)
        os.makedirs(base_dir, exist_ok=True)
    assert util.find_categories('differentdir') == ['cat', 'dog']


def test_find_categories_binary(imagedata):
    """It should put a not_X category first."""
    for category in ['animal', 'not_animal']:
        base_dir = os.path.join(imagedata.IMAGE_DIRECTORY, 'all', category)
        os.makedirs(base_dir, exist_ok=True)
    assert util.find_categories() == ['not_animal', 'animal']


def test_num_samples(imagedata):
    """It should return the number of samples in a category."""
    for category in ['cat', 'dog']:
        base_dir = os.path.join(imagedata.IMAGE_DIRECTORY, 'train', category)
        os.makedirs(base_dir, exist_ok=True)
        for i in range(3):
            with open(os.path.join(base_dir, '{}.txt'.format(i)), 'w') as f:
                f.write('')
    
    assert util.num_samples() == 3