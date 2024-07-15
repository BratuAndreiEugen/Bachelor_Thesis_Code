from domain.model import AudioItem
from infrastructure.cloud.buckets import BucketClient
from infrastructure.repository.audioitem_repository import AudioItemRepository
from infrastructure.repository.ml_repository import ModelPredictionRepository
from service.utils.note_utils import NoteUtils


class AudioItemService:
    def __init__(self, audioitem_repo: AudioItemRepository, bucket_client: BucketClient, note_utils: NoteUtils, config):
        self.__audioitem_repo = audioitem_repo
        self.__bucket_client = bucket_client
        self.__note_utils = note_utils
        self.__config = config

    def get_all_audioitems(self):
        return self.__audioitem_repo.get_all()

    def get_audioitems_for_user(self, user_id):
        return self.__audioitem_repo.get_audioitems_for_user(user_id=user_id)

    def get_audioitem_by_id(self, item_id):
        return self.__audioitem_repo.get(item_id)

    def save_audiofile_locally_for_audioitem(self, item_id):
        item_object = self.__audioitem_repo.get(item_id)
        s3_key = "audio/" + item_object.path_to_wav_file.split("/")[-1]
        local_path = self.__config["Project_Root"] + "\\bucket_temp\\" + item_object.path_to_wav_file.split("/")[-1]
        self.__bucket_client.download_file(s3_key, local_download_path=local_path)
        return item_object, local_path

    def add_audioitem(self, audio_file, poster_id, suggested_instrument, suggested_genre):
        """
        Adds audio item to db with user given input
        :param audio_file:
        :param poster_id:
        :param suggested_instrument:
        :param suggested_genre:
        :return:
        """
        audio_file_bucket_url = self.__bucket_client.upload_file(audio_file, file_type="audio", local=True,
                                                                 ignore_type_for_dir=True)
        audio_filename = audio_file
        added_item = self.__audioitem_repo.add(
            AudioItem(name=audio_filename.split("\\")[-1].split(".")[0], path_to_wav_file=audio_file_bucket_url,
                      path_to_video_file="",
                      instrument_majority_vote=suggested_instrument, genre_majority_vote=suggested_genre,
                      user_id=poster_id))
        return added_item

    def generate_video(self, audio_file, added_item: AudioItem, instrument_pred, genre_pred):
        """
        Generates a video with the notes of the audio file. Finds the potential pentatonic and normal scale and saves them to db
        :param audio_file:
        :param added_item:
        :return:
        """
        pentatonic_scale, scale, local_video_path = self.__note_utils.do_it_all(audio_file, fft_window_seconds=0.25,
                                                                                fps=30, instrument_pred=instrument_pred,
                                                                                genre_pred=genre_pred)
        video_file_bucket_url = self.__bucket_client.upload_file(local_video_path, file_type="video", local=True,
                                                                 ignore_type_for_dir=True)
        added_item.pentatonic_scale = ", ".join(note[0] for note in pentatonic_scale)
        added_item.scale = ", ".join(note[0] for note in scale)
        added_item.path_to_video_file = video_file_bucket_url
        return self.__audioitem_repo.update(added_item)
