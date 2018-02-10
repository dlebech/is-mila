"""Utility for generating training and validation data.
"""
import math
import os
import random
import shutil

from .. import config


def find_categories():
    """Return a list of all categories that are currently available."""
    return os.listdir(os.path.join(config.IMAGE_DIRECTORY, 'all'))


def split_categories(categories):
    """Split the given categories. Each category is assumed to be the name of
    a subdirectory with images.
    
    """
    # Ensure that the train and validation directories are empty beforehand
    shutil.rmtree(os.path.join(config.IMAGE_DIRECTORY, 'train'), ignore_errors=True)
    shutil.rmtree(os.path.join(config.IMAGE_DIRECTORY, 'validation'), ignore_errors=True)

    for category in categories:
        # Find all images for the category and shuffle the order of the images
        images = os.listdir(os.path.join(config.IMAGE_DIRECTORY, 'all', category))
        random.shuffle(images)

        # Split the images
        split_point = math.ceil(len(images) * 0.8) if len(images) >= 100 else len(images)
        train_images = images[:split_point]
        validation_images = images[split_point:]

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


def run():
    categories = find_categories()
    split_categories(categories)