from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Define the base
Base = declarative_base()


# User table
class User(Base):
    __tablename__ = 'Users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    uname = Column(String, nullable=False)
    email = Column(String)
    password_hash = Column(String, nullable=False)
    audio_items = relationship('AudioItem', back_populates='user')
    instrument_votes = relationship('InstrumentVote', back_populates='user')
    genre_votes = relationship('GenreVote', back_populates='user')


# AudioItems table
class AudioItem(Base):
    __tablename__ = 'AudioItems'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    path_to_wav_file = Column(String)
    path_to_video_file = Column(String)
    scale = Column(String)
    pentatonic_scale = Column(String)
    instrument_majority_vote = Column(String)
    genre_majority_vote = Column(String)
    user_id = Column(Integer, ForeignKey('Users.id'))
    user = relationship('User', back_populates='audio_items')
    instrument_votes = relationship('InstrumentVote', back_populates='audio_item')
    genre_votes = relationship('GenreVote', back_populates='audio_item')


# InstrumentVotes table
class InstrumentVote(Base):
    __tablename__ = 'InstrumentVotes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument = Column(String, nullable=False)
    audio_id = Column(Integer, ForeignKey('AudioItems.id'))
    user_id = Column(Integer, ForeignKey('Users.id'))
    audio_item = relationship('AudioItem', back_populates='instrument_votes')
    user = relationship('User', back_populates='instrument_votes')


# GenreVotes table
class GenreVote(Base):
    __tablename__ = 'GenreVotes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    genre = Column(String, nullable=False)
    audio_id = Column(Integer, ForeignKey('AudioItems.id'))
    user_id = Column(Integer, ForeignKey('Users.id'))
    audio_item = relationship('AudioItem', back_populates='genre_votes')
    user = relationship('User', back_populates='genre_votes')


# Datasets table
class Dataset(Base):
    __tablename__ = 'Datasets'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    nr_classes = Column(Integer)
    type = Column(String)
    models = relationship('Model', back_populates='train_dataset')


# Models table
class Model(Base):
    __tablename__ = 'Models'

    name = Column(String, primary_key=True)
    train_acc = Column(Float)
    validation_acc = Column(Float)
    input_length_seconds = Column(Float)
    input_shape = Column(String)
    feature_description = Column(String)
    model_type = Column(Integer)
    train_dataset_id = Column(Integer, ForeignKey('Datasets.id'))
    train_dataset = relationship('Dataset', back_populates='models')
    model_predictions = relationship('ModelPrediction', back_populates='model')


# ModelPredictions table
class ModelPrediction(Base):
    __tablename__ = 'ModelPredictions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    audio_id = Column(Integer, ForeignKey('AudioItems.id'))
    model_name = Column(String, ForeignKey('Models.name'))
    majority_class = Column(String)
    prediction_string = Column(String)
    audio_item = relationship('AudioItem')
    model = relationship('Model', back_populates='model_predictions')
