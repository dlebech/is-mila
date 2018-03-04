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
        './anotherplace/mymodeldir',
        '--imagesize',
        '111,222',
        '--batchsize',
        '456'])
    simple_mock.assert_called_with((111, 222), 123, 456, './anotherplace/mymodeldir')


def test_main_train_mobilenet_defaults(mocker):
    """It should run the mobilenet training with default parameters."""
    simple_mock = mocker.patch('mila.train.mobilenet.train')
    cli.main(['train', 'mobilenet'])
    simple_mock.assert_called_with(10, 32, './output/mobilenet')


def test_main_predict_defaults(mocker):
    """It should predict with default parameters."""
    simple_mock = mocker.patch('mila.predict.predict')
    cli.main(['predict', 'images/cat.jpg', 'output/mymodeldir'])
    simple_mock.assert_called_with('images/cat.jpg', 'output/mymodeldir')


def test_main_explore(mocker):
    """It should launch quiver"""
    mocker.patch('mila.predict.load', return_value=('FAKE_MODEL', {'classes': ['cat', 'dog']}))
    mocker.patch('tempfile.mkdtemp', return_value='FAKE_TMP_DIR')
    quiver_mock = mocker.patch('quiver_engine.server.launch')
    cli.main(['explore', 'images/all/cat', 'output/mymodeldir'])
    quiver_mock.assert_called_with(
        'FAKE_MODEL',
        classes=['cat', 'dog'],
        input_folder='images/all/cat',
        temp_folder='FAKE_TMP_DIR',
        std=[255, 255, 255])