import json
import logging
from logging.handlers import RotatingFileHandler

from flask import Flask, request, jsonify

from infrastructure.cloud.buckets import AWSBucketClient
from infrastructure.repository.audioitem_repository import AudioItemRepository
from infrastructure.repository.base import DBClient
from infrastructure.repository.ml_repository import DatasetRepository, ModelRepository, ModelPredictionRepository
from infrastructure.repository.user_repository import UserRepository
from infrastructure.repository.vote_repository import GenreVoteRepository, InstrumentVoteRepository
from service.audioitem_service import AudioItemService
from service.ml_layer.runners import RunnerSetUp
from service.prediction_service import PredictionService
from service.user_service import UserService
from service.utils.note_utils import NoteUtils
from web.audioitem_controller import audioitem_bp
from web.user_controller import user_bp
from web.web_utils.api_doc import ApiDoc

app = Flask(__name__)

def setup_logger():
    # Create a logger
    logger = logging.getLogger('my_flask_app')
    logger.setLevel(logging.DEBUG)

    # Create a file handler that logs debug and higher level messages
    file_handler = RotatingFileHandler('app.log', maxBytes=10240, backupCount=10)
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler for output to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Add the logger to the Flask app
    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.DEBUG)

setup_logger()

with open("./config.json") as json_data_file:
    general_config = json.load(json_data_file)

app.config["PROJECT_ROOT"] = general_config["Project_Root"]

# Bucket_Client
bucket_client = AWSBucketClient(general_config)

# Db_Client
db_client = DBClient(general_config)

# Repos
user_repo = UserRepository(general_config, db_client)
audioitem_repo = AudioItemRepository(general_config, db_client)
dataset_repo = DatasetRepository(general_config, db_client)
model_repo = ModelRepository(general_config, db_client)
pred_repo = ModelPredictionRepository(general_config, db_client)
ivote_repo = InstrumentVoteRepository(general_config, db_client)
gvote_repo = GenreVoteRepository(general_config, db_client)

# Machine Learning Integration Classes
runner_setup = RunnerSetUp(general_config, model_repo=model_repo)
ml_runners = runner_setup.create_model_runners("\\ml_models")

# Utils
note_utils = NoteUtils(image_folder=general_config["Project_Root"]+"\\service\\utils\\content", resolution=(general_config["Video_Width"], general_config["Video_Height"]))

# Services
user_service = UserService(user_repo=user_repo)
app.config["USER_SERVICE"] = user_service
audioitem_service = AudioItemService(audioitem_repo=audioitem_repo, bucket_client=bucket_client, note_utils=note_utils, config=general_config)
app.config["AUDIOITEM_SERVICE"] = audioitem_service

prediction_service = PredictionService(prediction_repo=pred_repo, best_feature_ml_runners=ml_runners)
app.config["PREDICTION_SERVICE"] = prediction_service

app.register_blueprint(user_bp, url_prefix="/users")
app.register_blueprint(audioitem_bp)


if __name__ == '__main__':
    ApiDoc.list_routes(app)
    app.run(debug=True, port=5000)
