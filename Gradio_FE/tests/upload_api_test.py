import json
import time

from api_clients.audioitem_api import AudioItemApi
from api_clients.user_api import UserClient
from dtos.dto_model import AudioItemDTO

with open("../config.json") as json_data_file:
    general_config = json.load(json_data_file)

time_now = str(time.time())
user_client = UserClient(base_url=general_config["API_HOSTNAME"] + ":" + general_config["API_PORT"] + "/users")

audioitem_client = AudioItemApi(base_url=general_config["API_HOSTNAME"] + ":" + general_config["API_PORT"])

typeclass = int(input("Ce clasificăm (1-inst, 2-genuri, 3-video): "))
if typeclass != 2:
    prefix = input("Alege prefix(irmas, philarmonia, kaggle): ")
while True:
    if typeclass == 1:
        user_in = int(input("Alege request (doar instrumente): "))
        try:
            if user_in == 0:
                response = audioitem_client.add_audioitem(3, "../wavfiles/sax_1.wav", "Saxophone", "Classical", prefix + "_svm",
                                                          "none", generate_video="no")
                print(response)
                print(response.json())
            if user_in == 1:
                response = audioitem_client.add_audioitem(3, "../wavfiles/sax_1.wav", "Saxophone", "Classical", prefix + "_rf",
                                                          "none", generate_video="no")
                print(response)
                print(response.json())
            if user_in == 2:
                response = audioitem_client.add_audioitem(3, "../wavfiles/sax_1.wav", "Saxophone", "Classical", prefix + "_knn",
                                                          "none", generate_video="no")
                print(response)
                print(response.json())
            if user_in == 3:
                response = audioitem_client.add_audioitem(3, "../wavfiles/sax_1.wav", "Saxophone", "Classical", prefix + "_cnng1",
                                                          "none", generate_video="no")
                print(response)
                print(response.json())
            if user_in == 4:
                response = audioitem_client.add_audioitem(3, "../wavfiles/sax_1.wav", "Saxophone", "Classical", prefix + "_rnng1",
                                                          "none", generate_video="no")
                print(response)
                print(response.json())
            if user_in == 5:
                response = audioitem_client.add_audioitem(3, "../wavfiles/sax_1.wav", "Saxophone", "Classical", prefix + "_resnet",
                                                          "none", generate_video="no")
                print(response)
                print(response.json())
            if user_in == 6:
                response = audioitem_client.add_audioitem(3, "../wavfiles/sax_1.wav", "Saxophone", "Classical", "kaggle_cnn_one",
                                                          "none", generate_video="no")
                print(response)
                print(response.json())
            if user_in == 7:
                response = audioitem_client.add_audioitem(3, "../wavfiles/sax_1.wav", "Saxophone", "Classical", "kaggle_rnn_one",
                                                          "none", generate_video="no")
                print(response)
                print(response.json())
            if user_in > 7:
                typeclass = int(input("Ce clasificăm (1-inst, 2-genuri, 3-video): "))
                if typeclass != 2:
                    prefix = input("Alege prefix(irmas, philarmonia, kaggle): ")
                continue
        except Exception as e:
            print(f"Eroare : {e}")
    if typeclass == 2:
        user_in = int(input("Alege request (doar genuri): "))
        try:
            if user_in == 0:
                response = audioitem_client.add_audioitem(3, "../wavfiles/sax_1.wav", "Saxophone", "Classical",
                                                          "none",
                                                          "gtzan_svm", generate_video="no")
                print(response)
                print(response.json())
            if user_in == 1:
                response = audioitem_client.add_audioitem(3, "../wavfiles/sax_1.wav", "Saxophone", "Classical",
                                                          "none",
                                                          "gtzan_rf", generate_video="no")
                print(response)
                print(response.json())
            if user_in == 2:
                response = audioitem_client.add_audioitem(3, "../wavfiles/sax_1.wav", "Saxophone", "Classical",
                                                          "none",
                                                          "gtzan_knn", generate_video="no")
                print(response)
                print(response.json())
            if user_in == 3:
                response = audioitem_client.add_audioitem(3, "../wavfiles/sax_1.wav", "Saxophone", "Classical",
                                                          "none",
                                                          "gtzan_cnng1", generate_video="no")
                print(response)
                print(response.json())
            if user_in == 4:
                response = audioitem_client.add_audioitem(3, "../wavfiles/sax_1.wav", "Saxophone", "Classical",
                                                          "none",
                                                          "gtzan_rnng1", generate_video="no")
                print(response)
                print(response.json())
            if user_in == 5:
                response = audioitem_client.add_audioitem(3, "../wavfiles/sax_1.wav", "Saxophone", "Classical",
                                                          "none",
                                                          "gtzan_resnet", generate_video="no")
                print(response)
                print(response.json())
            if user_in > 5:
                typeclass = int(input("Ce clasificăm (1-inst, 2-genuri, 3-video): "))
                if typeclass != 2:
                    prefix = input("Alege prefix(irmas, philarmonia, kaggle): ")
                continue
        except Exception as e:
            print(f"Eroare : {e}")
    if typeclass == 3:
        user_in = int(input("Alege request (ambele clas pt. video): "))
        try:
            if user_in == 0:
                response = audioitem_client.add_audioitem(3, "../wavfiles/sax_1.wav", "Saxophone", "Classical",
                                                          prefix + "_svm",
                                                          "gtzan_svm", generate_video="yes")
                print(response)
                print(response.json())
            if user_in == 1:
                response = audioitem_client.add_audioitem(3, "../wavfiles/sax_1.wav", "Saxophone", "Classical",
                                                          prefix + "_rf",
                                                          "gtzan_resnet", generate_video="yes")
                print(response)
                print(response.json())
            if user_in > 1:
                typeclass = int(input("Ce clasificăm (1-inst, 2-genuri, 3-video): "))
                if typeclass != 2:
                    prefix = input("Alege prefix(irmas, philarmonia, kaggle): ")
                continue
        except Exception as e:
            print(f"Eroare : {e}")