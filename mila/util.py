"""Common utility function."""
import logging
import os
import typing

import tensorflow as tf
from sklearn.utils import class_weight

from . import config


logger = logging.getLogger(__name__)


def find_categories(from_dir="all") -> typing.List[str]:
    """Return a list of all categories that are currently available."""
    dirs = os.listdir(os.path.join(config.IMAGE_DIRECTORY, from_dir))

    # A hack to make sure that the list is sorted, but not_X categories end up
    # in the beginning of the list.
    dirs.sort(key=lambda x: "aaaaaaaa" if x.startswith("not_") else x)

    return dirs


def num_samples(from_dir="train"):
    """Find the number of image samples in the given folder."""
    categories = find_categories(from_dir=from_dir)
    return sum(
        len(os.listdir(os.path.join(config.IMAGE_DIRECTORY, from_dir, category)))
        for category in categories
    )


def compute_class_weight(category_mapping: dict, from_dir="train") -> dict:
    """Compute the class weights for samples in the given directory."""
    categories = find_categories(from_dir=from_dir)
    y = [
        category_mapping[category]
        for category in categories
        for _ in range(
            len(os.listdir(os.path.join(config.IMAGE_DIRECTORY, from_dir, category)))
        )
    ]

    class_weights = class_weight.compute_class_weight(
        "balanced", list(set(category_mapping.values())), y
    )

    return {i: v for i, v in enumerate(class_weights)}


def image_preprocessing_fun(trainer):
    if trainer == config.TRAINER_SIMPLE:
        return lambda x: x / 255.0

    if trainer == config.TRAINER_MOBILENET:
        return tf.keras.applications.mobilenet_v2.preprocess_input

    logger.warning("Unknown trainer {}".format(trainer))

    # Fall back to the same preprocessing as the simple trainer
    return lambda x: x / 255.0
