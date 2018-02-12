"""Tests for the Flickr image preparation module.
"""
import os
import shutil
from unittest.mock import MagicMock

import pytest

from mila.prepare import flickr


class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


@pytest.fixture
def config_change(mocker):
    config_mock = mocker.patch('mila.prepare.flickr.config')
    config_mock.IMAGE_DIRECTORY = './tests/images'
    config_mock.FLICKR_API_KEY = 'xxxx'
    yield
    shutil.rmtree('./tests/images', ignore_errors=True)


def create_session_get_callable(data):
    class MockGet(MagicMock):
        async def __call__(self, *args, **kwargs):
            resp_mock = MagicMock()
            resp_mock.json.return_value = data
            resp_mock.content = data
            return resp_mock
    return MockGet


def test_merge_params(config_change):    
    """It should merge the default params into the given params."""
    params = {
        'method': 'flickr.people.getPhotos',
    }
    assert flickr.add_default_params(params) == {
        'method': 'flickr.people.getPhotos',
        'api_key': 'xxxx',
        'format': 'json',
        'nojsoncallback': 1
    }

    params = {
        'media': 'photos'
    }
    assert flickr.add_default_params(params) == {
        'media': 'photos',
        'api_key': 'xxxx',
        'format': 'json',
        'nojsoncallback': 1
    }


def test_create_flickr_url():
    """It should return the correct url"""
    assert flickr.create_flickr_url({
        'farm': 4,
        'server': 123,
        'id': 456,
        'secret': 789
    }, 'm') == 'https://farm4.staticflickr.com/123/456_789_m.jpg'


@pytest.mark.asyncio
async def test_get_public_photos_empty(mocker):
    """It should return an empty list when the object returned is empty."""
    get = create_session_get_callable({})
    mocker.patch.object(flickr.session, 'get', new_callable=get)
    assert await flickr.get_public_photos('abcd', 1, 10) == []


@pytest.mark.asyncio
async def test_get_public_photos_non_empty(mocker):
    """It should return the photos in the photo list."""
    get = create_session_get_callable({'photos': {'photo': [{'id': 123}]}})
    mocker.patch.object(flickr.session, 'get', new_callable=get)
    assert await flickr.get_public_photos('abcd', 1, 10) == [
        {'id': 123}
    ]


@pytest.mark.asyncio
async def test_process_photo_matches_tag(mocker, config_change):
    """It should write the photo to the tags location."""
    mocker.patch.object(flickr.session, 'get', new_callable=create_session_get_callable(b'testdata'))
    await flickr.process_photo({
        'id': 111,
        'farm': 222,
        'server': 333,
        'secret': 555,
        'tags': 'cat dog unity'
    }, ['cat'])
    expected_filename = './tests/images/all/cat/111.jpg'
    assert os.path.exists(expected_filename)
    with open(expected_filename) as f:
        assert f.read() == 'testdata'


@pytest.mark.asyncio
async def test_process_photo_no_match_one_category(mocker, config_change):
    """It should write the photo to the not_tag location."""
    mocker.patch.object(flickr.session, 'get', new_callable=create_session_get_callable(b'testdata'))
    await flickr.process_photo({
        'id': 111,
        'farm': 222,
        'server': 333,
        'secret': 555,
        'tags': 'only dog here'
    }, ['cat'])
    expected_filename = './tests/images/all/not_cat/111.jpg'
    assert os.path.exists(expected_filename)
    with open(expected_filename) as f:
        assert f.read() == 'testdata'


@pytest.mark.asyncio
async def test_process_photo_no_match_multiple_categories(mocker):
    """It should not write any photos if there are more than one target tag."""
    mocker.patch.object(flickr.session, 'get', new_callable=create_session_get_callable(b'testdata'))
    l = mocker.patch('mila.prepare.flickr.logger')
    await flickr.process_photo({
        'id': 111,
        'farm': 222,
        'server': 333,
        'secret': 555,
        'tags': 'no valid tags here'
    }, ['cat', 'dog'])
    l.info.assert_any_call('Flickr photo is not valid for classification: https://farm222.staticflickr.com/333/111_555_z.jpg')


@pytest.mark.asyncio
async def test_fetch_photos_stop_immediately(mocker):
    """It should stop when there are no more photos"""
    p = mocker.patch('mila.prepare.flickr.get_public_photos', new_callable=AsyncMock)
    p.return_value = []
    mocker.patch('mila.prepare.flickr.process_photo', new_callable=AsyncMock)

    await flickr.fetch_photos('abcd', ['cat'], 10)

    # Assert that it stopped after the first batch
    p.assert_any_call('abcd', 1, 10)
    with pytest.raises(AssertionError):
        p.assert_any_call('abcd', 2, 10)


@pytest.mark.asyncio
async def test_fetch_photos_stop_after_paging(mocker):
    """It should fetch next page when limit has not been reached."""
    p = mocker.patch('mila.prepare.flickr.get_public_photos', new_callable=AsyncMock)
    p.side_effect = [[{}], [{}], []]
    mocker.patch('mila.prepare.flickr.process_photo', new_callable=AsyncMock)

    await flickr.fetch_photos('abcd', ['cat'], 10)

    # Assert that it stopped after the first two batches
    p.assert_any_call('abcd', 1, 10)
    p.assert_any_call('abcd', 2, 10)
    p.assert_any_call('abcd', 3, 10)
    with pytest.raises(AssertionError):
        p.assert_any_call('abcd', 4, 10)


@pytest.mark.asyncio
async def test_fetch_photos_stop_at_limit(mocker):
    """It should stop when the limit is reached"""
    p = mocker.patch('mila.prepare.flickr.get_public_photos', new_callable=AsyncMock)
    p.side_effect = [[{}], [{}], []]
    mocker.patch('mila.prepare.flickr.process_photo', new_callable=AsyncMock)

    # Note the limit of 1
    await flickr.fetch_photos('abcd', ['cat'], 1)

    # Assert that it stopped after the first batch
    p.assert_any_call('abcd', 1, 10)
    with pytest.raises(AssertionError):
        p.assert_any_call('abcd', 2, 10)