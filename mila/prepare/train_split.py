"""Utility for generating training and validation data.
"""
import logging
import math
import os
import random
import shutil

from .. import config, util


logger = logging.getLogger(__name__)


def split_categories(categories, equal_splits=True):
    """Split the given categories. Each category is assumed to be the name of
    a subdirectory with images.

    """
    # Ensure that the train and validation directories are empty beforehand
    shutil.rmtree(os.path.join(config.IMAGE_DIRECTORY, 'train'), ignore_errors=True)
    shutil.rmtree(os.path.join(config.IMAGE_DIRECTORY, 'validation'), ignore_errors=True)

    min_images = min(
        len(os.listdir(os.path.join(config.IMAGE_DIRECTORY, 'all', category)))
        for category in categories
    )

    if min_images < 100:
        logger.info('At least one class has less than 100 samples, will not create validation set')

    if equal_splits:
        logger.info('Splitting categories equally, maximum {} images'.format(min_images))

    for category in categories:
        logger.info('Saving images for category {}'.format(category))

        # Find all images for the category and shuffle the order of the images
        images = os.listdir(os.path.join(config.IMAGE_DIRECTORY, 'all', category))
        random.shuffle(images)

        if equal_splits:
            images = images[:min_images]

        train_images = images
        validation_images = []

        # All categories need at least 100 samples to split, otherwise we use
        # everything for training.
        if min_images >= 100:
            # Split the images
            split_point = math.ceil(len(images) * 0.8)
            train_images = images[:split_point]
            validation_images = images[split_point:]

        logger.info('{} trains and {} validation images'
                    .format(len(train_images), len(validation_images)))

        # Ensure the target directories are created.
        os.makedirs(os.path.join(config.IMAGE_DIRECTORY, 'train', category), exist_ok=True)
        os.makedirs(os.path.join(config.IMAGE_DIRECTORY, 'validation', category), exist_ok=True)

        # Copy the images
        for train_image in train_images:
            shutil.copy(
                os.path.join(config.IMAGE_DIRECTORY, 'all', category, train_image),
                os.path.join(config.IMAGE_DIRECTORY, 'train', category, train_image))
        for validation_image in validation_images:
            shutil.copy(
                os.path.join(config.IMAGE_DIRECTORY, 'all', category, validation_image),
                os.path.join(config.IMAGE_DIRECTORY, 'validation', category, validation_image))


def run(equal_splits=True):
    """Run the train/validation split command"""
    categories = util.find_categories()
    split_categories(categories, equal_splits=equal_splits)
