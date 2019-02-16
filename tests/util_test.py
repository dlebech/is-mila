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


@pytest.fixture
def cats_and_dogs(imagedata):
    # Create 2 cats and 4 dogs
    base_dir = os.path.join(imagedata.IMAGE_DIRECTORY, 'train', 'cat')
    os.makedirs(base_dir, exist_ok=True)
    for i in range(2):
        with open(os.path.join(base_dir, '{}.txt'.format(i)), 'w') as f:
            f.write('')

    base_dir = os.path.join(imagedata.IMAGE_DIRECTORY, 'train', 'dog')
    os.makedirs(base_dir, exist_ok=True)
    for i in range(4):
        with open(os.path.join(base_dir, '{}.txt'.format(i)), 'w') as f:
            f.write('')



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


def test_num_samples(cats_and_dogs):
    """It should return the number of samples."""
    # 6 images in total
    assert util.num_samples() == 6


def test_compute_class_weight(cats_and_dogs):
    """It should compute class weights for cats and dogs."""
    assert util.compute_class_weight({'cat': 0, 'dog': 1}) == {
        0: 1.5,
        1: 0.75
    }


def test_compute_class_weight_reverse(cats_and_dogs):
    """It should accept a different class mapping"""
    assert util.compute_class_weight({'dog': 0, 'cat': 1}) == {
        1: 1.5,
        0: 0.75
    }