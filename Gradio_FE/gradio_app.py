import json
import os

import soundfile as sf
import gradio
import gradio as gr
import requests

from api_clients.audioitem_api import AudioItemApi
from api_clients.prediction_api import ModelPredictionApi
from api_clients.user_api import UserClient
from utils.feature_extractor import FeatureExtractor
from utils.video import VideoUtils

with open("config.json") as json_data_file:
    general_config = json.load(json_data_file)

# Instantiate the API clients
user_client = UserClient(base_url=general_config["API_HOSTNAME"] + ":" + general_config["API_PORT"] + "/users")
audio_client = AudioItemApi(base_url=general_config["API_HOSTNAME"] + ":" + general_config["API_PORT"])
prediction_client = ModelPredictionApi(base_url=general_config["API_HOSTNAME"] + ":" + general_config["API_PORT"])

session = {}
generate_video_flag = False
feature_extractor = FeatureExtractor()


# Function to handle login
def login(username, password):
    response = user_client.login(username, password)
    if response.status_code == 200:
        session["current_user_email"] = response.json()["current_user_email"]
        session["current_user_id"] = response.json()["current_user_id"]
        session["current_user_name"] = response.json()["current_user_name"]
        return "Login successful! Session started. "
    return "Login failed! Please check your credentials."


def upload_audio(audio_file, instrument, genre, instrument_model=None, genre_model=None, generate_video=False,
                 response_elements=None):
    if response_elements is None:
        response_elements = {}
    if audio_file is None:
        return "No file uploaded!"
    if instrument_model is None:
        instrument_model = "none"
    if genre_model is None:
        genre_model = "none"
    if generate_video is True:
        generate_video = "yes"
    if generate_video is False:
        generate_video = "no"

    # Save the uploaded file temporarily
    # temp_file_path = os.path.join("/tmp", audio_file.name)
    # with open(temp_file_path, "wb") as f:
    #     f.write(audio_file.read())

    try:
        user_id = session["current_user_id"]
    except:
        user_id = '4'
    # Call integrate_audioitem function
    response = audio_client.add_audioitem(user_id=user_id, temp_file_path=audio_file,
                                          instrument_from_user=instrument, genre_from_user=genre,
                                          selected_instrument_model=instrument_model,
                                          selected_genre_model=genre_model, generate_video=generate_video)

    # Clean up the temporary file
    # os.remove(temp_file_path)
    resp_dict = response.json()
    # updates = {}
    # for key, value in response_elements.items():
    #     if resp_dict[key]:
    #         updates[key] = gr.update(visible=bool(value), value=value)
    if not resp_dict["path_to_video_file"]:
        resp_dict["path_to_video_file"] = "Not existent"
    return resp_dict["path_to_wav_file"], resp_dict["path_to_video_file"], " ".join(
        resp_dict["instrument_prediction_list"]), " ".join(resp_dict["genre_prediction_list"]), resp_dict["id"]


def update_checkbox_state(checkbox_value):
    global generate_video_flag
    generate_video_flag = checkbox_value


# Function to fetch and display AudioItem details
def fetch_audioitem_details(item_id):
    audioitem_response = audio_client.get_audioitem_by_id(item_id=item_id)
    predictions_response = prediction_client.get_all_predictions_for_audioitem(audioitem_id=item_id)

    if audioitem_response.status_code == 200:
        audioitem_data = audioitem_response.json().get('item', {})
    else:
        audioitem_data = {"error": "Failed to fetch audio item details"}

    if predictions_response.status_code == 200:
        predictions_data = predictions_response.json().get('predictions', [])
    else:
        predictions_data = [{"error": "Failed to fetch predictions"}]

    audioitem_details = {
        "id": audioitem_data.get('id', 'N/A'),
        "name": audioitem_data.get('name', 'N/A'),
        "path_to_wav_file": audioitem_data.get('path_to_wav_file', 'N/A'),
        "path_to_video_file": audioitem_data.get('path_to_video_file', 'N/A'),
        "instrument_majority_vote": audioitem_data.get('instrument_majority_vote', 'N/A'),
        "genre_majority_vote": audioitem_data.get('genre_majority_vote', 'N/A')
    }
    for key, value in audioitem_details.items():
        if not value:
            audioitem_details[key] = "N/A"

    predictions_df = [
        [
            pred.get('model_name', 'N/A'),
            pred.get('majority_class', 'N/A'),
            pred.get('prediction_string', 'N/A')
        ]
        for pred in predictions_data
    ]

    return (
        audioitem_details["id"], audioitem_details["name"], audioitem_details["path_to_wav_file"],
        audioitem_details["path_to_video_file"],
        audioitem_details["instrument_majority_vote"], audioitem_details["genre_majority_vote"],
        predictions_df
    )


def add_new_prediction_and_reload(item_id, model_name):
    add_response = prediction_client.add_prediction_for_item_model(item_id, model_name)
    predictions_response = prediction_client.get_all_predictions_for_audioitem(audioitem_id=item_id)

    if add_response.status_code == 200 and predictions_response.status_code == 200:
        predictions_data = predictions_response.json().get('predictions', [])
    else:
        predictions_data = [{"error": "Failed add new prediction"}]
    predictions_df = [
        [
            pred.get('model_name', 'N/A'),
            pred.get('majority_class', 'N/A'),
            pred.get('prediction_string', 'N/A')
        ]
        for pred in predictions_data
    ]
    return predictions_df


# Login interface
def login_interface():
    print("Loaded login_interface")
    with gr.Blocks() as login_page:
        gr.Markdown("# Login if you want your contribution to be credited (otherwise we'll never know)")
        username = gr.Textbox(label="Username", placeholder="Your username")
        password = gr.Textbox(label="Password", type="password", placeholder="Your password")
        status = gr.Markdown(label="status")
        login_button = gr.Button("Login")
        login_button.click(login, inputs=[username, password], outputs=status)
    return login_page


# Upload interface
def upload_interface():
    print("Loaded upload_interface")
    with gr.Blocks() as upload_page:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("# Upload Audio")
                audio_file = gr.File(label="Audio File")
                instrument = gr.Dropdown(label="What instrument do you think is in the recording?",
                                         choices=["Banjo", "Bass Clarinet", "Cello", "Clarinet", "English Horn",
                                                  "Flute", "French Horn", "Acoustic Guitar", "Electric Guitar",
                                                  "Mandolin", "Oboe", "Organ", "Piano", "Saxophone", "Trombone",
                                                  "Trumpet", "Tuba", "Viola", "Violin", "Bass Drum", "Double Bass",
                                                  "Hi-Hat", "Snare Drum"])
                genre = gr.Dropdown(label="What genre of music would the recording fit in?",
                                    choices=["Blues", "Classical", "Country", "Disco", "Hiphop", "Jazz", "Metal", "Pop",
                                             "Reggae", "Rock"])

                generate_video = gr.Checkbox(label="Generate video (!! It takes a long time)")
                generate_video.change(update_checkbox_state, inputs=generate_video)
                instrument_model = gr.Dropdown(label="Which ML model should be used for instrument classification?",
                                               choices=["irmas_cnng1", "philarmonia_cnng1", "kaggle_cnn_one",
                                                        "irmas_rnng1",
                                                        "philarmonia_rnng1",
                                                        "kaggle_rnn_one", "irmas_svm", "philarmonia_svm", "kaggle_svm",
                                                        "irmas_rf",
                                                        "philarmonia_rf", "kaggle_rf", "irmas_knn", "philarmonia_knn",
                                                        "kaggle_knn"])
                genre_model = gr.Dropdown(label="Which ML model should be used for genre classification?",
                                          choices=["gtzan_cnng1", "gtzan_rnng1", "gtzan_svm", "gtzan_rf", "gtzan_knn"])
                upload_button = gr.Button("Upload")

            with gr.Column(scale=1):
                response_elements = {
                    "path_to_wav_file": gr.Audio(visible=True),
                    "path_to_video_file": gr.Video(visible=True),
                    "instrument_prediction_list": gr.Markdown(visible=True, label="Instrument predicts"),
                    "genre_prediction_list": gr.Markdown(visible=True, label="Genre predicts"),
                    "item_id": gr.Number(visible=True, interactive=False, label="Generated ID")
                }
        upload_button.click(upload_audio,
                            inputs=[audio_file, instrument, genre, instrument_model, genre_model, generate_video],
                            outputs=list(response_elements.values()))
    return upload_page


# Combining all interfaces
def item_interface():
    print("Loaded item_interface")
    with gr.Blocks() as item_page:
        with gr.Row():
            with gr.Column(scale=1):
                item_id_input = gr.Number(label="Audio Item ID")
            with gr.Column(scale=1):
                fetch_button = gr.Button("Find")
        with gr.Row():
            with gr.Column(scale=1):
                audioitem_id = gr.Textbox(visible=False)
                audioitem_name = gr.Textbox(label="Name")
                wav_file = gr.Audio()
                video_file = gr.Video()
                instrument_majority_vote = gr.Markdown(label="User given instrument")
                genre_majority_vote = gr.Markdown(label="User given genre")
            with gr.Column(scale=1):
                predictions_output = gr.Dataframe(headers=["Model Name", "Majority Class", "Prediction String"])
                new_model = gr.Dropdown(label="Try other models",
                                        choices=["irmas_cnng1", "philarmonia_cnng1", "gtzan_cnng1", "kaggle_cnn_one",
                                                 "irmas_rnng1",
                                                 "philarmonia_rnng1", "gtzan_rnng1",
                                                 "kaggle_rnn_one", "irmas_svm", "philarmonia_svm", "gtzan_svm",
                                                 "kaggle_svm",
                                                 "irmas_rf",
                                                 "philarmonia_rf", "gtzan_rf", "kaggle_rf", "irmas_knn",
                                                 "philarmonia_knn", "gtzan_knn",
                                                                    "kaggle_knn"])
                new_predict_button = gr.Button("Go")

        new_predict_button.click(add_new_prediction_and_reload, inputs=[audioitem_id, new_model],
                                 outputs=predictions_output)
        fetch_button.click(
            fetch_audioitem_details,
            inputs=item_id_input,
            outputs=[
                audioitem_id, audioitem_name, wav_file, video_file,
                instrument_majority_vote, genre_majority_vote,
                predictions_output
            ]
        )
    return item_page


def populate_gallery(audio_input, step, sr, feature_name):
    og_sampling_rate = audio_input[0]
    og_y = audio_input[1]
    sf.write("temp_wav/temp.wav", og_y, og_sampling_rate)
    image_list = feature_extractor.make_image_list_for_input("temp_wav/temp.wav", feature_name=feature_name, sr=sr,
                                                             step=step)
    return image_list


def feature_interface():
    print("Loaded item_interface")
    with gr.Blocks() as feature_page:
        with gr.Row():
            with gr.Column(scale=1):
                audio_data = gr.Audio()
                granularity = gr.Number(label="Length in seconds for the prediction input")
                sampling_rate = gr.Number(label="Resample rate")
                feature_choice = gr.Dropdown(label="Choose feature",
                                             choices=["Mel Spectrogram", "20 MFCC", "13 MFCC", "Chroma STFT", "RMS",
                                                      "Spectral Centroid", "Spectral Bandwidth", "Spectral Rolloff",
                                                      "Zero Crossing Rate", "Harmony"])
                feature_button = gr.Button("Generate features")
            with gr.Column(scale=1):
                image_gallery = gr.Gallery()
        feature_button.click(populate_gallery, inputs=[audio_data, granularity, sampling_rate, feature_choice],
                             outputs=image_gallery)

    return feature_page


def app_interface():
    with gr.Blocks() as app:
        with gr.Tab("Upload"):
            upload_interface()
        with gr.Tab("Audio Item Search"):
            item_interface()
        with gr.Tab("Feature Visualisation"):
            feature_interface()
        with gr.Tab("Login"):
            login_interface()
    return app


app = app_interface()
app.launch()
