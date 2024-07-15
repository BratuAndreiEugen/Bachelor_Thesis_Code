class AudioItemDTO:
    def __init__(self, id=None, name=None, path_to_wav_file=None, path_to_video_file=None, scale=None,
                 pentatonic_scale=None,
                 instrument_majority_vote=None, genre_majority_vote=None, inst_pred_list=None, genre_pred_list=None):
        if inst_pred_list is None:
            inst_pred_list = []
        if genre_pred_list is None:
            genre_pred_list = []
        if scale is None:
            scale = []
        if pentatonic_scale is None:
            pentatonic_scale = []

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

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data.get('id'),
            name=data.get('name'),
            path_to_wav_file=data.get('path_to_wav_file'),
            path_to_video_file=data.get('path_to_video_file'),
            scale=data.get('scale', []),
            pentatonic_scale=data.get('pentatonic_scale', []),
            instrument_majority_vote=data.get('instrument_majority_vote'),
            genre_majority_vote=data.get('genre_majority_vote'),
            inst_pred_list=data.get('instrument_prediction_list', []),
            genre_pred_list=data.get('genre_prediction_list', [])
        )


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

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data.get('id'),
            audio_id=data.get('audio_id'),
            model_name=data.get('model_name'),
            majority_class=data.get('majority_class'),
            prediction_string=data.get('prediction_string')
        )
