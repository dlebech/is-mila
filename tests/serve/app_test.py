"""Tests for the serving app."""
import os
import shutil
import tempfile

import pytest

from mila import config
from mila.train import simple
from mila.serve.app import app

tmp = tempfile.mkdtemp(prefix='mila_tests_')
output_dir = os.path.join(tmp, 'output')
image_dir = os.path.join(tmp, 'images')


@pytest.fixture(scope='module')
def modeldata():
    train_dir1 = os.path.join(image_dir, 'train', 'a')
    train_dir2 = os.path.join(image_dir, 'train', 'b')
    val_dir1 = os.path.join(image_dir, 'validation', 'a')
    val_dir2 = os.path.join(image_dir, 'validation', 'b')
    os.makedirs(train_dir1), os.makedirs(train_dir2)
    os.makedirs(val_dir1), os.makedirs(val_dir2)
    shutil.copyfile('./tests/data/A.png', os.path.join(train_dir1, 'A.png'))
    shutil.copyfile('./tests/data/A.png', os.path.join(val_dir1, 'A.png'))
    shutil.copyfile('./tests/data/B.png', os.path.join(train_dir2, 'B.png'))
    shutil.copyfile('./tests/data/B.png', os.path.join(val_dir2, 'B.png'))
    prev_image_dir = config.IMAGE_DIRECTORY
    prev_output_dir = config.OUTPUT_DIRECTORY
    config.IMAGE_DIRECTORY = image_dir
    config.OUTPUT_DIRECTORY = output_dir
    simple.train((30, 30), 1, 32, '{}/ab'.format(output_dir), None, False)
    yield
    # Re-set these config settings manually since the pytest-mock lib does not
    # work on a module level. XXX: Perhaps find a better way.
    config.IMAGE_DIRECTORY = prev_image_dir
    config.OUTPUT_DIRECTORY = prev_output_dir
    shutil.rmtree(tmp)


@pytest.fixture
def configmock(mocker):
    config_mock = mocker.patch('mila.serve.app.config')
    config_mock.IMAGE_DIRECTORY = image_dir
    config_mock.OUTPUT_DIRECTORY = output_dir


def test_index_get(modeldata, configmock):
    """It should return a list of the available models."""
    _, res = app.test_client.get('/model')
    assert res.status == 200
    assert res.json.get('models') == ['ab']


def test_model_get_404(modeldata, configmock):
    """It should return 404 for non-existing models."""
    _, res = app.test_client.get('/model/doesnotexist')
    assert res.status == 404
    assert res.json.get('error') == 'Model not found: doesnotexist'


def test_model_get_success(modeldata, configmock):
    """It should return 200 for existing models."""
    _, res = app.test_client.get('/model/ab', headers={
        'Accept': 'application/json'
    })
    assert res.status == 200
    assert res.json.get('network', {}) == {
        'batch_size': None,
        'image': {
            'rows': 30,
            'cols': 30,
            'color_channels': 3
        }
    }
    assert res.json.get('meta', {}).get('classes') == ['a', 'b']
    assert res.json.get('meta', {}).get('latest_logs', {}).get('acc') >= 0
    assert res.json.get('meta', {}).get('latest_logs', {}).get('loss') >= 0
    assert res.json.get('meta', {}).get('latest_logs', {}).get('val_acc') >= 0
    assert res.json.get('meta', {}).get('latest_logs', {}).get('val_loss') >= 0


def test_model_get_prediction(modeldata, configmock):
    """It should return a prediction."""
    f = open(os.path.join(image_dir, 'train', 'a', 'A.png'), 'rb')
    _, res = app.test_client.post('/model/ab/prediction', data=f)
    assert res.status == 200
    assert len(res.json) == 1
    assert 'a' in res.json[0]
    assert 'b' in res.json[0]
