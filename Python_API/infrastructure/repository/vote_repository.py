from domain.model import InstrumentVote, GenreVote
from infrastructure.repository.base import BaseRepository


class InstrumentVoteRepository(BaseRepository):
    def __init__(self, config, db_client):
        super().__init__(config, db_client)
        self._entity_class = InstrumentVote


class GenreVoteRepository(BaseRepository):
    def __init__(self, config, db_client):
        super().__init__(config, db_client)
        self._entity_class = GenreVote
