"""Flickr-based image preparation"""
import json
import logging
import os

import requests
import requests_threads

from .. import config

FLICKR_REST = 'https://api.flickr.com/services/rest/'

logger = logging.getLogger(__name__)
session = requests_threads.AsyncSession(n=10)

def add_default_params(params : dict) -> dict:
    params.update({
        'api_key': config.FLICKR_API_KEY,
        'format': 'json',
        'nojsoncallback': 1
    })
    return params


def create_flickr_url(photo, size):
    return 'https://farm{farm}.staticflickr.com/{server}/{id}_{secret}_{size}.jpg'.format(size=size, **photo)


async def get_public_photos(user_id: str, page: int, per_page: int):
    params = add_default_params({
        'method': 'flickr.people.getPhotos',
        'user_id': user_id,
        'content_type': 1,
        'media': 'photos',
        'page': page,
        'per_page': per_page,
        'extras': 'tags'
    })

    resp = await session.get(FLICKR_REST, params=params)

    # The photo list in JSON is at { 'photos': 'photo': [] }}
    return resp.json().get('photos', {}).get('photo', [])


async def process_photo(photo: dict, tags: list):
    """Process a single photo, as returned by the Flickr search API.
    """
    url = create_flickr_url(photo, 'z')
    
    # Check to see if the photo has the tags we searched for.
    # If the photo matches a tag, put the photo in the appropriate tag-named
    # folder.
    # If the photo does _not_ match a tag, put it in the not_tag folder if we
    # are doing binary classification (if there is only a single tag available)
    valid_tags = set(tags).intersection(photo.get('tags', '').split(' '))
    filename = ''
    if valid_tags:
        filename += '{}/all/{}/{}.jpg'.format(config.IMAGE_DIRECTORY, valid_tags.pop(), photo['id'])
    elif len(tags) == 1:
        filename += '{}/all/not_{}/{}.jpg'.format(config.IMAGE_DIRECTORY, tags[0], photo['id'])

    # Download the file if it is valid
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logger.debug('Downloading Flickr photo {}'.format(url))
        resp = await session.get(url, stream=True)

        with open(filename, 'wb') as f:
            f.write(resp.content)
    else:
        logger.info('Flickr photo is not valid for classification: {}'.format(url))

async def fetch_photos(user_id: str, tags: list, limit: int):
    found_photos = 0
    page, per_page = 1, 10

    futures = []

    while True:
        logger.info('Fetching {} photos, batch {}'.format(per_page, page))
        photos = await get_public_photos(user_id, page, per_page)
        page += 1
        found_photos += len(photos)

        for photo in photos:
            futures.append(process_photo(photo, tags))

        for future in futures:
            await future

        futures = []

        if not photos or found_photos >= limit:
            break

    logger.info('Done fetching photos')
        

def run(user_id, tags, limit):
    async def _run():
        await fetch_photos(user_id, tags.split(','), limit)
    session.run(_run)