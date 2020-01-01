"""Custom callbacks."""
import json
import logging

from tensorflow.keras.callbacks import Callback


class ModelMetadata(Callback):
    def __init__(self, filename, categories, trainer):
        super().__init__()
        
        self.filename = filename
        self.categories = categories
        self.logger = logging.getLogger(__name__)
        self.trainer = trainer

    def on_epoch_end(self, epoch, logs=None):
        # Save some extra model metadata so we will know later what a prediction
        # means.
        logs = logs or {}
        logs = {k: float(v) for k, v in logs.items()}
        with open(self.filename, 'w') as f:
            json.dump({
                'classes': self.categories,
                'latest_logs': dict(**logs),
                'trainer': self.trainer
            }, f)

