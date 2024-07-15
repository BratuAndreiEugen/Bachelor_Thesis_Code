from domain.model import AudioItem
from infrastructure.repository.base import BaseRepository


class AudioItemRepository(BaseRepository):
    def __init__(self, config, db_client):
        super().__init__(config, db_client)
        self._entity_class = AudioItem

    def get_audioitems_for_user(self, user_id):
        return self.session.query(AudioItem).filter_by(user_id=user_id)
