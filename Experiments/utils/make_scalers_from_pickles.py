import json
import pickle

with open("../instrument_pickles/conv.p", 'rb') as handle:
    conv_data = pickle.load(handle)
    scaler = conv_data
    print(conv_data.min)
    print(conv_data.max)

print(scaler)
cnn_dict = {"min" : conv_data.min, "max": conv_data.max}

json_file_path = 'kaggle_cnn_one_scaler.json'

# Open the file in write mode and use json.dump to write the dictionary to the file
with open(json_file_path, 'w') as json_file:
    json.dump(cnn_dict, json_file)

with open("../instrument_pickles/recurrent.p", 'rb') as handle:
    rec_data = pickle.load(handle)
    print(rec_data.min)
    print(rec_data.max)

rnn_dict = {"min" : rec_data.min, "max": rec_data.max}

json_file_path = 'kaggle_rnn_one_scaler.json'

# Open the file in write mode and use json.dump to write the dictionary to the file
with open(json_file_path, 'w') as json_file:
    json.dump(rnn_dict, json_file)