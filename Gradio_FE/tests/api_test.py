import json
import time

from api_clients.audioitem_api import AudioItemApi
from api_clients.user_api import UserClient
from dtos.dto_model import AudioItemDTO

with open("../config.json") as json_data_file:
    general_config = json.load(json_data_file)

time_now = str(time.time())
user_client = UserClient(base_url=general_config["API_HOSTNAME"] + ":" + general_config["API_PORT"] + "/users")

# register_response = user_client.register("andrei", "andrei_bratu@yahoo.ro", "parola") # id = 4
# print(register_response)
# print(register_response.json())
# print(register_response.json()["current_user_id"])

login_response = user_client.login("andrei", "parola")

print(login_response)
print(login_response.json())

audioitem_client = AudioItemApi(base_url=general_config["API_HOSTNAME"] + ":" + general_config["API_PORT"])

# response = audioitem_client.add_audioitem(3, "wavfiles/sax_1.wav", "Saxophone", "Classical", "irmas_svm", "gtzan_rf")
# print(response)
# print(response.json())

# response = audioitem_client.add_audioitem(3, "../wavfiles/violin.mp3", "Violin", "Classical", "irmas_svm", "gtzan_rf", generate_video="yes")
# print(response)
# print(response.json())
#
# audioitem_dto = AudioItemDTO.from_dict(response.json())
# print(audioitem_dto)

response = audioitem_client.get_audioitem_by_id(83)
print(response)
print(response.json())