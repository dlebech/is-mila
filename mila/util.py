"""Common utility function."""
import os
import typing

from . import config


def find_categories(from_dir='all') -> typing.List[str]:
    """Return a list of all categories that are currently available."""
    dirs = os.listdir(os.path.join(config.IMAGE_DIRECTORY, from_dir))

    # A hack to make sure that the list is sorted, but not_X categories end up
    # in the beginning of the list.
    dirs.sort(key=lambda x: 'aaaaaaaa' if x.startswith('not_') else x)

    return dirs


def num_samples(from_dir='train'):
    """Find the mimium number of image samples in the givem folder."""
    categories = find_categories(from_dir=from_dir)
    return min(
        len(os.listdir(os.path.join(config.IMAGE_DIRECTORY, from_dir, category)))
        for category in categories
    )
