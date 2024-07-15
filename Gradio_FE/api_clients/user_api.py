import requests


class UserClient:
    def __init__(self, base_url):
        self.__base_url = base_url

    def login(self, username, password):
        request_body = {
            "username": username,
            "password": password
        }
        response = requests.post(self.__base_url + "/login", json=request_body)
        return response

    def register(self, username, email, password):
        request_body = {
            "username": username,
            "email": email,
            "password": password
        }
        response = requests.post(self.__base_url + "/register", json=request_body)
        return response