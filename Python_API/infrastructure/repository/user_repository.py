from domain.model import User
from infrastructure.repository.base import BaseRepository


class UserRepository(BaseRepository):
    def __init__(self, config, db_client):
        super().__init__(config, db_client)
        self._entity_class = User

    def get_user_by_username(self, uname) -> User:
        return self.session.query(User).filter_by(uname=uname).first()