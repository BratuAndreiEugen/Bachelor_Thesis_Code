from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import pyodbc


class DBClient:
    def __init__(self, config):
        self._config = config
        if self._config["Windows_Auth"] == "yes":
            db_url = 'mssql+pyodbc://@' + self._config["AWS_DB_URL"] + '/music_ai_db?driver=ODBC+Driver+17+for+SQL+Server'
        else:
            db_url = 'mssql+pyodbc://' + self._config["AWS_DB_Username"] + ":" + self._config["AWS_DB_Pass"] + "@" + \
                     self._config["AWS_DB_URL"] + ":" + self._config[
                         "AWS_DB_Port"] + "/music_ai_db?driver=ODBC+Driver+17+for+SQL+Server"
        print(db_url)
        self.__engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.__engine)

    def get_engine(self):
        return self.__engine


class BaseRepository:
    def __init__(self, config, db_client):
        self._config = config
        self.session = db_client.Session()
        self._entity_class = None

    def add(self, entity):
        self.session.add(entity)
        self.session.commit()
        self.session.refresh(entity)
        return entity

    def get(self, entity_id):
        return self.session.query(self._entity_class).get(entity_id)

    def update(self, entity):
        self.session.commit()
        self.session.refresh(entity)
        return entity

    def delete(self, entity):
        self.session.delete(entity)
        self.session.commit()

    def get_all(self):
        return self.session.query(self._entity_class).all()
