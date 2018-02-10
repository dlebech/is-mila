import argparse

import pytest

from ismila import cli


def test_main_prepare_flickr(mocker):
    """It should run the Flickr preparer with default limit."""
    flickr_mock = mocker.patch('ismila.prepare.flickr.run')

    cli.main(['prepare', 'flickr', '--user', 'abcd', '--tags', 'cat'])

    flickr_mock.assert_called_once()
    flickr_mock.assert_called_with('abcd', 'cat', 10)


def test_main_prepare_flickr_limit(mocker):
    """It should run the Flickr preparer with specified limit."""
    flickr_mock = mocker.patch('ismila.prepare.flickr.run')
    cli.main(['prepare', 'flickr', '--user', 'abcd', '--tags', 'cat', '--limit', '123'])

    flickr_mock.assert_called_once()
    flickr_mock.assert_called_with('abcd', 'cat', 123)


def test_main_prepare_test_split(mocker):
    """It should run the Test split run function."""
    split_mock = mocker.patch('ismila.prepare.train_split.run')
    cli.main(['prepare', 'traindata'])
    split_mock.assert_called_once()