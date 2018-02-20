"""Custom callbacks."""
import json
import logging
import os

from keras.callbacks import Callback


class ModelMetadata(Callback):
    def __init__(self, filename, categories):
        super().__init__()
        
        self.filename = filename
        self.categories = categories
        self.logger = logging.getLogger(__name__)

    def on_epoch_end(self, epoch, logs=None):
        # Save some extra model metadata so we will know later what a prediction
        # means.
        logs = logs or {}
        with open(self.filename, 'w') as f:
            json.dump({
                'classes': self.categories,
                'latest_logs': dict(**logs)
            }, f)

