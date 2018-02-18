"""Tests for the train_split module.
"""
import os
import shutil
import tempfile

import pytest

from mila.prepare import train_split


@pytest.fixture
def num_images():
    return 1


@pytest.fixture
def categories():
    return ['cat', 'dog']


@pytest.fixture
def imagedata(mocker, num_images, categories):
    tmp = tempfile.mkdtemp(prefix='mila_tests_')
    config_mock = mocker.patch('mila.prepare.train_split.config')
    config_mock.IMAGE_DIRECTORY = os.path.join(tmp, 'images')
    for i, category in enumerate(categories):
        base_dir = os.path.join(config_mock.IMAGE_DIRECTORY, 'all', category)
        os.makedirs(base_dir, exist_ok=True)
        for j in range(num_images + i):
            with open(os.path.join(base_dir, '{}.txt'.format(j)), 'w') as f:
                f.write('')
    yield config_mock.IMAGE_DIRECTORY
    shutil.rmtree(tmp)


@pytest.mark.parametrize('num_images', [5])
def test_split_categories_less_than_hundred(imagedata, categories):
    """It should not split the data if there are less than 100 images."""
    train_split.split_categories(categories)
    assert len(os.listdir(os.path.join(imagedata, 'train/cat'))) == 5
    assert len(os.listdir(os.path.join(imagedata, 'train/dog'))) == 5
    assert len(os.listdir(os.path.join(imagedata, 'validation/cat'))) == 0
    assert len(os.listdir(os.path.join(imagedata, 'validation/dog'))) == 0


@pytest.mark.parametrize('num_images', [100])
def test_split_categories_more_than_hundred(imagedata, categories):
    """It should split the data if there are more than 100 images."""
    train_split.split_categories(categories)
    assert len(os.listdir(os.path.join(imagedata, 'train/cat'))) == 80
    assert len(os.listdir(os.path.join(imagedata, 'train/dog'))) == 80
    assert len(os.listdir(os.path.join(imagedata, 'validation/cat'))) == 20
    assert len(os.listdir(os.path.join(imagedata, 'validation/dog'))) == 20


@pytest.mark.parametrize('num_images', [5])
def test_split_categories_non_equal_splits(imagedata, categories):
    """It should not split the data if there are less than 100 images."""
    train_split.split_categories(categories, equal_splits=False)
    assert len(os.listdir(os.path.join(imagedata, 'train/cat'))) == 5
    assert len(os.listdir(os.path.join(imagedata, 'train/dog'))) == 6
    assert len(os.listdir(os.path.join(imagedata, 'validation/cat'))) == 0
    assert len(os.listdir(os.path.join(imagedata, 'validation/dog'))) == 0
