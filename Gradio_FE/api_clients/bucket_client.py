import requests


class BucketClient:
    def __init__(self):
        pass

    @staticmethod
    def send_audio(file_path, url):
        files = {'audio': open(file_path, 'rb')}
        response = requests.post(url, files=files)
        return response.json()