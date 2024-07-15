from collections import Counter
from typing import List

from bidict import bidict

from domain.model import AudioItem, ModelPrediction
from infrastructure.repository.ml_repository import ModelPredictionRepository
from service.ml_layer.runners import MLRunner


class PredictionService:
    def __init__(self, prediction_repo : ModelPredictionRepository, best_feature_ml_runners : List[MLRunner]):
        self.__best_feature_ml_runners = best_feature_ml_runners
        self.__prediction_repo = prediction_repo
        self.__dataset_dict = {
            "irmas_novoice": bidict({'cel': 0, 'cla': 1, 'flu': 2, 'gac': 3,
                                     'gel': 4, 'org': 5, 'pia': 6, 'sax': 7,
                                     'tru': 8, 'vio': 9}),
            "philarmonia": bidict(
                {'banjo': 0, 'bass_clarinet': 1, 'bassoon': 2, 'cello': 3, 'clarinet': 4, 'contrabassoon': 5,
                 'cor_anglais': 6, 'double_bass': 7, 'flute': 8, 'french_horn': 9, 'guitar': 10,
                 'mandolin': 11, 'oboe': 12, 'percussion': 13, 'saxophone': 14, 'trombone': 15,
                 'trumpet': 16, 'tuba': 17, 'viola': 18, 'violin': 19}),
            "kaggle_instruments": bidict({'Acoustic_guitar': 0, 'Bass_drum': 1, 'Cello': 2, 'Clarinet': 3,
                                          'Double_bass': 4, 'Flute': 5, 'Hi-hat': 6, 'Saxophone': 7,
                                          'Snare_drum': 8, 'Violin_or_fiddle': 9}),
            "gtzan": bidict({
                'blues': 0, 'classical': 1, 'country': 2, 'disco': 3,
                'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7,
                'reggae': 8, 'rock': 9})
        }

    def get_predictions_for_model(self, model_name):
        return self.__prediction_repo.get_predictions_for_model(model_name)

    def get_predictions_for_audioitem(self, audioitem_id):
        return self.__prediction_repo.get_predictions_for_audioitem(audioitem_id)

    def predict_audioitem(self, audio_file, added_item : AudioItem, model_name=None):
        """
        Adds a prediction for the audio item with the given model
        :param audio_file:
        :param added_item:
        :param model_name:
        :return:
        """
        if model_name:
            instrument_found_prediction = []
            dataset_name = None
            for runner in self.__best_feature_ml_runners:
                if runner.get_model_db().name == model_name:
                    features = runner.extract_features(audio_file)
                    prediction = runner.predict_for_model(features)
                    print(prediction)
                    dataset_name = runner.get_model_db().train_dataset.name
                    instrument_found_prediction = prediction

            if dataset_name:
                ds_dict = self.__dataset_dict[dataset_name]
                strings = [ds_dict.inverse[item] for item in instrument_found_prediction]

            counter = Counter(strings)

            # Find the string with the highest count
            most_common_class, count_ins = counter.most_common(1)[0]

            return strings, self.__prediction_repo.add(ModelPrediction(audio_id=added_item.id, model_name=model_name, majority_class=most_common_class, prediction_string=",".join(strings)))
