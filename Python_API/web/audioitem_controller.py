import json
import os
import pickle
import tempfile
import uuid

import librosa
import numpy as np
from flask import Blueprint, request, jsonify, current_app, Response
from scipy.io import wavfile
from werkzeug.utils import secure_filename

from service.utils.folder_utils import FolderUtils
from web.dtos.dto_model import AudioItemDTO, ModelPredictionDTO

audioitem_bp = Blueprint('audioitem', __name__)


class AudioItemRoutes:
    @staticmethod
    @audioitem_bp.route('/audioitems', methods=['POST'])
    def integrate_item():
        if 'audio' not in request.files:
            return jsonify({
                "error": "Invalid request. No Audio File in request."
            }), 400

        audio_file = request.files['audio']
        if 'json' not in request.form:
            return jsonify({
                "error": "Invalid request. No JSON in request."
            }), 400

        data = json.loads(request.form['json'])
        if not data or 'user_id' not in data:
            return jsonify({
                "error": "Invalid request. Please provide user_id in JSON."
            }), 400

        poster_id = int(data["user_id"])
        instrument_model = data["selected_instrument_model"]
        genre_model = data["selected_genre_model"]
        generate_video_flag = data["generate_video"]
        instrument = data["suggested_instrument"]
        genre = data["suggested_genre"]

        print(audio_file.filename)
        filename = current_app.config["PROJECT_ROOT"] + "\\audio_temp\\" + audio_file.filename
        audio_file.save(filename)
        try:
            # Save the audio item in DB ( saves audio file in Bucket too )
            added_audioitem = current_app.config["AUDIOITEM_SERVICE"].add_audioitem(filename, poster_id, instrument,
                                                                                    genre)

            # Save and retrun predict for instrument classification
            if instrument_model == "none":
                inst_pred_list = None
            else:
                inst_pred_list, inst_pred_obj = current_app.config["PREDICTION_SERVICE"].predict_audioitem(filename,
                                                                                                           added_audioitem,
                                                                                                           model_name=instrument_model)

            # Save and return predict for genre classification
            if genre_model == "none":
                genre_pred_list = None
            else:
                genre_pred_list, genre_pred_obj = current_app.config["PREDICTION_SERVICE"].predict_audioitem(filename,
                                                                                                             added_audioitem,
                                                                                                             model_name=genre_model)

            # Generate video if asked to do so. WARNING : IT WILL TAKE TIME
            if generate_video_flag == "yes":
                added_audioitem = current_app.config["AUDIOITEM_SERVICE"].generate_video(filename, added_audioitem,
                                                                                         inst_pred_list,
                                                                                         genre_pred_list)

        except Exception as e:
            print(f"Eroare in controller: {e}")
            return jsonify({
                "error": "Internal Server Error"
            }), 500
        finally:
            FolderUtils.clear_folder(current_app.config["PROJECT_ROOT"] + "\\audio_temp")

        # Item DTO
        audioitem_dto = AudioItemDTO(added_audioitem.id, added_audioitem.name, added_audioitem.path_to_wav_file,
                                     added_audioitem.path_to_video_file, added_audioitem.scale,
                                     added_audioitem.pentatonic_scale,
                                     added_audioitem.instrument_majority_vote, added_audioitem.genre_majority_vote,
                                     inst_pred_list, genre_pred_list)

        return jsonify(audioitem_dto.to_dict()), 201

    @staticmethod
    @audioitem_bp.route('/audioitems/<int:item_id>/models/<string:model_name>/predictions', methods=['POST'])
    def add_prediction_for_item(item_id, model_name):
        try:
            item_object, local_path = current_app.config["AUDIOITEM_SERVICE"].save_audiofile_locally_for_audioitem(item_id)
            _, prediction_object = current_app.config["PREDICTION_SERVICE"].predict_audioitem(local_path, item_object, model_name=model_name)
            dto = ModelPredictionDTO(prediction_object.id, prediction_object.audio_id, prediction_object.model_name, prediction_object.majority_class, prediction_object.prediction_string)
        except Exception as e:
            print(f"Eroare in controller: {e}")
            return jsonify({
                "error": "Internal Server Error"
            }), 500
        finally:
            FolderUtils.clear_folder(current_app.config["PROJECT_ROOT"] + "\\bucket_temp")
        return jsonify({
            "prediction" : dto.to_dict()
        }), 200
    @staticmethod
    @audioitem_bp.route('/audioitems', methods=['GET'])
    def get_all_items():
        try:
            audioitems = current_app.config["AUDIOITEM_SERVICE"].get_all_audioitems()
            audioitem_dtos = []
            for item in audioitems:
                dto = AudioItemDTO(item.id, item.name, item.path_to_wav_file,
                                   item.path_to_video_file, item.scale, item.pentatonic_scale,
                                   item.instrument_majority_vote, item.genre_majority_vote)
                audioitem_dtos.append(dto.to_dict())
        except Exception as e:
            print(f"Eroare in controller: {e}")
            return jsonify({
                "error": "Internal Server Error"
            }), 500
        return jsonify({
            "items": audioitem_dtos
        }), 200

    @staticmethod
    @audioitem_bp.route('/audioitems/<int:item_id>', methods=['GET'])
    def get_item_by_id(item_id):
        try:
            audioitem = current_app.config["AUDIOITEM_SERVICE"].get_audioitem_by_id(item_id)
            dto = AudioItemDTO(audioitem.id, audioitem.name, audioitem.path_to_wav_file,
                               audioitem.path_to_video_file, audioitem.scale, audioitem.pentatonic_scale,
                               audioitem.instrument_majority_vote, audioitem.genre_majority_vote)

        except Exception as e:
            print(f"Eroare in controller: {e}")
            return jsonify({
                "error": "Internal Server Error"
            }), 500
        return jsonify({
            "item" : dto.to_dict()
        }), 200
    
    
    @staticmethod
    @audioitem_bp.route('/users/<int:user_id>/audioitems', methods=['GET'])
    def get_all_items_for_user(user_id):
        try:
            audioitems = current_app.config["AUDIOITEM_SERVICE"].get_audioitems_for_user(user_id=user_id)
            audioitem_dtos = []
            for item in audioitems:
                dto = AudioItemDTO(item.id, item.name, item.path_to_wav_file,
                                   item.path_to_video_file, item.scale, item.pentatonic_scale,
                                   item.instrument_majority_vote, item.genre_majority_vote)
                audioitem_dtos.append(dto.to_dict())
        except Exception as e:
            print(f"Eroare in controller: {e}")
            return jsonify({
                "error": "Internal Server Error"
            }), 500
        return jsonify({
            "items": audioitem_dtos
        }), 200

    @staticmethod
    @audioitem_bp.route('/models/<string:model_name>/predictions', methods=['GET'])
    def get_predictions_for_model(model_name):
        try:
            predictions = current_app.config["PREDICTION_SERVICE"].get_predictions_for_model(model_name=model_name)
            prediction_dtos = []
            for item in predictions:
                dto = ModelPredictionDTO(item.id, item.audio_id, item.model_name, item.majority_class, item.prediction_string)
                prediction_dtos.append(dto.to_dict())
        except Exception as e:
            print(f"Eroare in controller: {e}")
            return jsonify({
                "error": "Internal Server Error"
            }), 500
        return jsonify({
            "predictions": prediction_dtos
        }), 200

    @staticmethod
    @audioitem_bp.route('/audioitems/<int:item_id>/predictions', methods=['GET'])
    def get_predictions_for_audioitem(item_id):
        try:
            predictions = current_app.config["PREDICTION_SERVICE"].get_predictions_for_audioitem(audioitem_id=item_id)
            prediction_dtos = []
            for item in predictions:
                dto = ModelPredictionDTO(item.id, item.audio_id, item.model_name, item.majority_class,
                                         item.prediction_string)
                prediction_dtos.append(dto.to_dict())
        except Exception as e:
            print(f"Eroare in controller: {e}")
            return jsonify({
                "error": "Internal Server Error"
            }), 500
        return jsonify({
            "predictions": prediction_dtos
        }), 200
