from domain.model import Dataset, Model, ModelPrediction
from infrastructure.repository.base import BaseRepository


class DatasetRepository(BaseRepository):
    def __init__(self, config, db_client):
        super().__init__(config, db_client)
        self._entity_class = Dataset


class ModelRepository(BaseRepository):
    def __init__(self, config, db_client):
        super().__init__(config, db_client)
        self._entity_class = Model


class ModelPredictionRepository(BaseRepository):
    def __init__(self, config, db_client):
        super().__init__(config, db_client)
        self._entity_class = ModelPrediction

    def get_predictions_for_model(self, model_name):
        return self.session.query(ModelPrediction).filter_by(model_name=model_name)

    def get_predictions_for_audioitem(self, audioitem_id):
        return self.session.query(ModelPrediction).filter_by(audio_id=audioitem_id)