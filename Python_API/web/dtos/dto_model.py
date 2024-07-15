class AudioItemDTO:
    def __init__(self, id, name, path_to_wav_file, path_to_video_file, scale, pentatonic_scale,
                 instrument_majority_vote, genre_majority_vote, inst_pred_list=None, genre_pred_list=None):
        if inst_pred_list is None:
            inst_pred_list = []
        if genre_pred_list is None:
            genre_pred_list = []
        self.id = id  # integer
        self.name = name
        self.path_to_wav_file = path_to_wav_file
        self.path_to_video_file = path_to_video_file
        self.scale = scale  # list of strings
        self.pentatonic_scale = pentatonic_scale  # list of strings
        self.instrument_majority_vote = instrument_majority_vote
        self.genre_majority_vote = genre_majority_vote
        self.instrument_prediction_list = inst_pred_list
        self.genre_prediction_list = genre_pred_list

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'path_to_wav_file': self.path_to_wav_file,
            'path_to_video_file': self.path_to_video_file,
            'scale': self.scale,
            'pentatonic_scale': self.pentatonic_scale,
            'instrument_majority_vote': self.instrument_majority_vote,
            'genre_majority_vote': self.genre_majority_vote,
            'instrument_prediction_list': self.instrument_prediction_list,
            'genre_prediction_list': self.genre_prediction_list
        }


class ModelPredictionDTO:
    def __init__(self, id, audio_id, model_name, majority_class, prediction_string):
        self.id = id
        self.audio_id = audio_id
        self.model_name = model_name
        self.majority_class = majority_class
        self.prediction_string = prediction_string

    def to_dict(self):
        return {
            'id': self.id,
            'audio_id': self.audio_id,
            'model_name': self.model_name,
            'majority_class': self.majority_class,
            'prediction_string': self.prediction_string
        }
