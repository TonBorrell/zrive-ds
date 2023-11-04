import os
import joblib
import numpy as np

from src.module_6.exceptions import PredictionException


MODEL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "bin/model.joblib")
)


class BasketModel:
    def __init__(self, logger):
        self.model = joblib.load(MODEL)
        self.logger = logger

    def predict(self, features: np.ndarray) -> np.ndarray:
        try:
            pred = self.model.predict(features)
            self.logger.info(f"Predict for {features}")
        except Exception as exception:
            self.logger.error("PredictionException: Error during model inference")
            raise PredictionException("Error during model inference") from exception
        return pred
