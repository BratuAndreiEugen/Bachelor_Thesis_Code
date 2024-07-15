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

prediction_client = ModelPredictionApi(base_url=general_config["API_HOSTNAME"] + ":" + general_config["API_PORT"])

response = prediction_client.add_prediction_for_item_model(84, "irmas_knn")
print(response)
print(response.json())

response = prediction_client.get_all_predictions_for_audioitem(84)
print(response)
print(response.json())