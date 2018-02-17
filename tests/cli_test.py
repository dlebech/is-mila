import argparse

import pytest

from mila import cli


def test_main_prepare_flickr(mocker):
    """It should run the Flickr preparer with default limit."""
    flickr_mock = mocker.patch('mila.prepare.flickr.run')

    cli.main(['prepare', 'flickr', '--user', 'abcd', '--tags', 'cat'])

    flickr_mock.assert_called_once()
    flickr_mock.assert_called_with('abcd', 'cat', 10)


def test_main_prepare_flickr_limit(mocker):
    """It should run the Flickr preparer with specified limit."""
    flickr_mock = mocker.patch('mila.prepare.flickr.run')
    cli.main(['prepare', 'flickr', '--user', 'abcd', '--tags', 'cat', '--limit', '123'])

    flickr_mock.assert_called_once()
    flickr_mock.assert_called_with('abcd', 'cat', 123)


def test_main_prepare_test_split(mocker):
    """It should run the Test split run function."""
    split_mock = mocker.patch('mila.prepare.train_split.run')
    cli.main(['prepare', 'traindata', '--equalsplits'])
    split_mock.assert_called_with(equal_splits=True)


def test_main_train_simple_defaults(mocker):
    """It should run the training with default parameters."""
    simple_mock = mocker.patch('mila.train.simple.train')
    cli.main(['train', 'simple'])
    simple_mock.assert_called_with((32, 32), 10, 32, './output/simple')


def test_main_train_simple_with_params(mocker):
    """It should run the training with given parameters."""
    simple_mock = mocker.patch('mila.train.simple.train')
    cli.main([
        'train',
        'simple',
        '--epochs',
        '123',
        '--outputdir',
        './tests',
        '--imagesize',
        '111,222',
        '--batchsize',
        '456'])
    simple_mock.assert_called_with((111, 222), 123, 456, './tests')


def test_main_predict_defaults(mocker):
    """It should run the training with default parameters."""
    simple_mock = mocker.patch('mila.predict.predict')
    cli.main(['predict', 'images/cat.jpg', 'output/'])
    simple_mock.assert_called_with('images/cat.jpg', 'output/')