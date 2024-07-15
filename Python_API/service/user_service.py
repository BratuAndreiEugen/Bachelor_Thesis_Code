from flask import current_app

from domain.model import User
from infrastructure.repository.user_repository import UserRepository
from service.utils.security import HashUtils


class UserService:
    def __init__(self, user_repo: UserRepository):
        self.__user_repo = user_repo

    def login(self, username, password):
        user_from_db = self.__user_repo.get_user_by_username(username)
        if user_from_db and HashUtils.verify_password(password, user_from_db.password_hash):
            return user_from_db
        else:
            return None

    def register(self, username, email, password):
        self.__user_repo.add(User(uname=username, email=email, password_hash=HashUtils.hash_password(password)))
