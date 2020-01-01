"""Flickr-based image preparation"""
import logging
import os

import requests_threads

from .. import config

FLICKR_REST = "https://api.flickr.com/services/rest/"

logger = logging.getLogger(__name__)
session = requests_threads.AsyncSession(n=10)


def add_default_params(params: dict) -> dict:
    """Add default Flickr paramaters to the given parameter dictionary."""
    params.update(
        {"api_key": config.FLICKR_API_KEY, "format": "json", "nojsoncallback": 1}
    )
    return params


def create_flickr_url(photo, size):
    """Create a Flickr image url based on the given photo object and size (in Flickr terms)."""
    # pylint: disable=C0301
    return "https://farm{farm}.staticflickr.com/{server}/{id}_{secret}_{size}.jpg".format(
        size=size, **photo
    )


async def get_public_photos(user_id: str, page: int, per_page: int):
    """Get publicly available photos for the given user and page, with a limit per page."""
    params = add_default_params(
        {
            "method": "flickr.people.getPhotos",
            "user_id": user_id,
            "content_type": 1,
            "media": "photos",
            "page": page,
            "per_page": per_page,
            "extras": "tags",
        }
    )

    resp = await session.get(FLICKR_REST, params=params)

    # The photo list in JSON is at { 'photos': 'photo': [] }}
    return resp.json().get("photos", {}).get("photo", [])


async def process_photo(photo: dict, tags: list):
    """Process a single photo, as returned by the Flickr search API. The
    photo is matched against the given tag list.

    If there is exactly one tag,
    the photo will always be put into a category, either corresponding to the
    matching or tag "not matching" the tag.

    If there are two or more tags in
    the list, photos that don't match any of the tags will be discarded.

    """
    url = create_flickr_url(photo, "z")

    # Check to see if the photo has the tags we searched for.
    # If the photo matches a tag, put the photo in the appropriate tag-named
    # folder.
    # If the photo does _not_ match a tag, put it in the not_tag folder if we
    # are doing binary classification (if there is only a single tag available)
    valid_tags = set(tags).intersection(photo.get("tags", "").split(" "))
    filename = ""
    if valid_tags:
        filename += "{}/all/{}/{}.jpg".format(
            config.IMAGE_DIRECTORY, valid_tags.pop(), photo["id"]
        )
    elif len(tags) == 1:
        filename += "{}/all/not_{}/{}.jpg".format(
            config.IMAGE_DIRECTORY, tags[0], photo["id"]
        )

    # Download the file if it is valid
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logger.debug("Downloading Flickr photo {}".format(url))
        resp = await session.get(url, stream=True)

        with open(filename, "wb") as f:
            f.write(resp.content)
    else:
        logger.info("Flickr photo is not valid for classification: {}".format(url))


async def fetch_photos(user_id: str, tags: list, limit: int):
    """Fetch photos for the given Flickr user ID and match them against the list of tags.

    The limit determines when to stop fetching images.

    """
    found_photos = 0
    page, per_page = 1, 10

    futures = []

    while True:
        logger.info("Fetching {} photos, batch {}".format(per_page, page))
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

    logger.info("Done fetching photos")


def run(user_id, tags, limit):
    """Runs the Flickr images preparation."""

    async def _run():
        await fetch_photos(user_id, tags.split(","), limit)

    session.run(_run)
