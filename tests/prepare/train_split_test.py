"""Tests for the train_split module.
"""
import os
import shutil

import pytest

from ismila.prepare import train_split


@pytest.fixture
def num_images():
    return 1


@pytest.fixture
def categories():
    return ['cat', 'dog']


@pytest.fixture
def imagedata(mocker, num_images, categories):
    config_mock = mocker.patch('ismila.prepare.train_split.config')
    config_mock.IMAGE_DIRECTORY = './tests/images'
    for category in categories:
        base_dir = os.path.join('./tests/images/all', category)
        os.makedirs(base_dir, exist_ok=True)
        for i in range(num_images):
            with open(os.path.join(base_dir, '{}.txt'.format(i)), 'w') as f:
                f.write('')
    yield
    shutil.rmtree('./tests/images')


def test_find_categories(imagedata):
    """It should find all the categories."""
    assert set(train_split.find_categories()) == set(['cat', 'dog'])


@pytest.mark.parametrize('num_images', [5])
def test_split_categories_less_than_hundred(imagedata, categories):
    """It should not split the data if there are less than 100 images."""
    train_split.split_categories(categories)
    assert len(os.listdir('./tests/images/train/cat')) == 5
    assert len(os.listdir('./tests/images/train/dog')) == 5
    assert len(os.listdir('./tests/images/validation/cat')) == 0
    assert len(os.listdir('./tests/images/validation/dog')) == 0


@pytest.mark.parametrize('num_images', [100])
def test_split_categories_more_than_hundred(imagedata, categories):
    """It should split the data if there are more than 100 images."""
    train_split.split_categories(categories)
    assert len(os.listdir('./tests/images/train/cat')) == 80
    assert len(os.listdir('./tests/images/train/dog')) == 80
    assert len(os.listdir('./tests/images/validation/cat')) == 20
    assert len(os.listdir('./tests/images/validation/dog')) == 20