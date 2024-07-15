import json

import librosa
import numpy as np
import requests
from scipy.io import wavfile


class AudioItemApi:
    def __init__(self, base_url):
        self.__base_url = base_url

    def get_all_audioitems(self):
        response = requests.get(self.__base_url + "/audioitems")
        return response

    def get_audioitems_for_user(self, user_id):
        response = requests.get(self.__base_url + f"/users/{user_id}/audioitems")
        return response

    def get_audioitem_by_id(self, item_id):
        response = requests.get(self.__base_url + f"/audioitems/{item_id}")
        return response

    def add_audioitem(self, user_id, temp_file_path, instrument_from_user, genre_from_user, selected_instrument_model, selected_genre_model, generate_video="no"):
        with open(temp_file_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        request_body = {
            "user_id": str(user_id),
            "selected_instrument_model": selected_instrument_model,
            "selected_genre_model": selected_genre_model,
            "suggested_instrument": instrument_from_user,
            "suggested_genre": genre_from_user,
            "generate_video": generate_video,
        }

        data = {'json': json.dumps(request_body)}
        print(temp_file_path)
        response = requests.post(self.__base_url + "/audioitems", files={'audio': (temp_file_path.split("/")[-1].split("\\")[-1], audio_bytes, 'audio/wav')}, data=data)
        return response