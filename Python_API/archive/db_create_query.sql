USE music_ai_db;
GO

CREATE TABLE Users (
    id INT PRIMARY KEY IDENTITY,
    uname VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    password_hash VARCHAR(255) NOT NULL
);

CREATE TABLE AudioItems (
    id INT PRIMARY KEY IDENTITY,
    name VARCHAR(255) NOT NULL,
    path_to_wav_file VARCHAR(255),
    path_to_video_file VARCHAR(255),
    scale VARCHAR(255),
    pentatonic_scale VARCHAR(255),
    instrument_majority_vote VARCHAR(255),
    genre_majority_vote VARCHAR(255),
    user_id INT,
    FOREIGN KEY (user_id) REFERENCES Users(id)
);

CREATE TABLE InstrumentVotes (
    id INT PRIMARY KEY IDENTITY,
    instrument VARCHAR(255) NOT NULL,
    audio_id INT,
    user_id INT,
    FOREIGN KEY (audio_id) REFERENCES AudioItems(id),
    FOREIGN KEY (user_id) REFERENCES Users(id)
);

CREATE TABLE GenreVotes (
    id INT PRIMARY KEY IDENTITY,
    genre VARCHAR(255) NOT NULL,
    audio_id INT,
    user_id INT,
    FOREIGN KEY (audio_id) REFERENCES AudioItems(id),
    FOREIGN KEY (user_id) REFERENCES Users(id)
);

CREATE TABLE Datasets (
    id INT PRIMARY KEY IDENTITY,
    name VARCHAR(255) NOT NULL,
    nr_classes INT,
    type VARCHAR(255)
);

CREATE TABLE Models (
    name VARCHAR(255) PRIMARY KEY,
    train_acc FLOAT,
    validation_acc FLOAT,
    train_dataset_id INT,
    input_length_seconds FLOAT,
    input_shape VARCHAR(255),
    feature_description VARCHAR(255),
    FOREIGN KEY (train_dataset_id) REFERENCES Datasets(id)
);

CREATE TABLE ModelPredictions (
    id INT PRIMARY KEY IDENTITY,
    audio_id INT,
    model_name VARCHAR(255),
    majority_class VARCHAR(255),
    FOREIGN KEY (audio_id) REFERENCES AudioItems(id),
    FOREIGN KEY (model_name) REFERENCES Models(name)
);
