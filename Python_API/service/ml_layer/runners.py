import json
import os
import pickle

import joblib
import numpy as np
from flask import current_app
from keras.src.saving import load_model

from domain import model
from infrastructure.repository.ml_repository import ModelRepository
from service.ml_layer.extractors import FeatureExtractor


class RunnerSetUp:
    def __init__(self, config, model_repo: ModelRepository):
        self.__config = config
        self.__model_repo = model_repo

    def create_model_runners(self, resource_path):
        resource_path = self.__config["Project_Root"] + resource_path
        runners = []
        print(self.__config["Model_Paths"]["Genre"])
        for key, value in self.__config["Model_Paths"]["Genre"].items():
            model_db_object = self.__model_repo.get(key)
            scaler_path = resource_path + "\\Genre\\" + self.__config["Scaler_Paths"]["Genre"][
                key.split("_")[0] + "_scaler"]
            runner = MLRunner(self.__config, model_db_object, model_class="Genre",
                              model_path=resource_path + "\\Genre\\" + value,
                              scaler_path=scaler_path)
            runners.append(runner)

        for key, value in self.__config["Model_Paths"]["Instrument"].items():
            model_db_object = self.__model_repo.get(key)
            scaler_path = resource_path + "\\Instrument\\" + self.__config["Scaler_Paths"]["Instrument"][
                key.split("_")[0] + "_scaler"]
            if key == "kaggle_cnn_one" or key == "kaggle_rnn_one":
                scaler_path = resource_path + "\\Instrument\\" + self.__config["Scaler_Paths"]["Instrument"][key+"_scaler"]
            runner = MLRunner(self.__config, model_db_object, model_class="Instrument",
                              model_path=resource_path + "\\Instrument\\" + value,
                              scaler_path=scaler_path)
            runners.append(runner)

        return runners


class MLRunner:
    def __init__(self, config, model_db_object: model.Model, model_class: str, model_path: str, scaler_path=None):
        self.__config = config
        self.__model_db_object = model_db_object
        if model_path.endswith(".joblib"):
            self.__model = joblib.load(model_path)
        else:
            self.__model = load_model(model_path)
        if self.__model_db_object.model_type == 0:
            self.__scaler = joblib.load(scaler_path)
        if self.__model_db_object.model_type == 3:
            print(scaler_path)
            with open(scaler_path) as json_data_file:
                self.__scaler = json.load(json_data_file)
        self.__input_shape = tuple(self.__model_db_object.input_shape.split(","))
        self.__model_class = model_class

    def predict_for_model(self, features):
        print(features.shape)
        prediction = self.__model.predict(features)
        feature_type = self.__model_db_object.model_type
        if feature_type != 0:
            prediction = [np.argmax(pred_line) for pred_line in prediction]
        return prediction

    def extract_features(self, file_path):
        feature_type = self.__model_db_object.model_type
        if feature_type == 0:
            return FeatureExtractor.extract_best_features(file_path, self.__scaler)
        elif feature_type == 1:
            return FeatureExtractor.extract_mfcc_librosa(file_path, cnn="cnn" in self.__model_db_object.name)
        elif feature_type == 2:
            return FeatureExtractor.extract_mel_specs(file_path)
        elif feature_type == 3:
            return FeatureExtractor.extract_mfcc_pysf(file_path, scaler=self.__scaler, cnn="cnn" in self.__model_db_object.name)

    def get_model_db(self):
        return self.__model_db_object

    def get_model_class(self):
        return self.__model_class
