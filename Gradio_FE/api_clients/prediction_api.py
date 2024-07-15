import json

import librosa
import numpy as np
import requests
from scipy.io import wavfile


class ModelPredictionApi:
    def __init__(self, base_url):
        self.__base_url = base_url

    def get_all_predictions_for_model(self, model_name="irmas_svm"):
        response = requests.get(self.__base_url + f"/models/{model_name}/predictions")
        return response

    def get_all_predictions_for_audioitem(self, audioitem_id):
        response = requests.get(self.__base_url + f"/audioitems/{audioitem_id}/predictions")
        return response

    def add_prediction_for_item_model(self, item_id, model_name):
        response = requests.post(self.__base_url + f"/audioitems/{item_id}/models/{model_name}/predictions")
        return response