import json
import time

from api_clients.audioitem_api import AudioItemApi
from api_clients.prediction_api import ModelPredictionApi
from api_clients.user_api import UserClient
from dtos.dto_model import AudioItemDTO

with open("../config.json") as json_data_file:
    general_config = json.load(json_data_file)

time_now = str(time.time())
user_client = UserClient(base_url=general_config["API_HOSTNAME"] + ":" + general_config["API_PORT"] + "/users")

audioitem_client = AudioItemApi(base_url=general_config["API_HOSTNAME"] + ":" + general_config["API_PORT"])

prediction_client = ModelPredictionApi(base_url=general_config["API_HOSTNAME"] + ":" + general_config["API_PORT"])

response = audioitem_client.add_audioitem(3, "../wavfiles/sax_1.wav", "Saxophone", "Classical",
                                          "irmas_svm",
                                          "gtzan_svm", generate_video="no")
print(response)
print(response.json())

response = audioitem_client.get_all_audioitems()
print(response)
print(response.json())

response = audioitem_client.get_audioitems_for_user(user_id=3)
print(response)
print(response.json())

response = prediction_client.get_all_predictions_for_model(model_name="irmas_svm")
print(response)
print(response.json())