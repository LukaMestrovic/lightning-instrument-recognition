# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:25:06 2023

@author: Francek
"""

from flask import Flask, request, render_template, jsonify
import hydra
from hydra import compose, initialize

import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append("../")
from src.data.data_preprocessing import extract_from_file
from src.models.lightning_modules import CNN_mel, CNN_modgd, CNN_pitch

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# global initialization
hydra.core.global_hydra.GlobalHydra.instance().clear()
initialize(version_base=None, config_path="../configs")
cfg = compose(config_name="config")

#hyper parameters
lr = cfg.training.learning_rate
num_classes = cfg.constants.num_classes

ALLOWED_EXTENSIONS = set(['wav'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

fusion_thresholds = np.load("../src/fusion/fusion_thresholds_mel_modgd_pitch_acc.npy")
# preprocessing
step_perc = 1.0 #koliko nam je step kad segmentiramo spektrogram - default 100%

# aggregation
aggregation_method = "s1"

# maps
predictions_map = {0: "cello", 1: "clarinet", 2: "flute", 3: "accoustic guitar", 4: "electric guitar", 5: "organ", 
                   6: "piano", 7: "saxophone", 8: "trumpet", 9: "violin", 10: "voice"}
label_map = {"cel": 0, "cla": 1, "flu": 2, "gac": 3, "gel": 4, "org": 5, 
             "pia": 6, "sax": 7, "tru": 8, "vio": 9, "voi": 10}

model_mel = CNN_mel.load_from_checkpoint('../models/cnn_mel.ckpt', lr=lr, num_labels=num_classes, map_location=device)
model_modgd = CNN_modgd.load_from_checkpoint('../models/cnn_modgd.ckpt', lr=lr, num_labels=num_classes, map_location=device)
model_pitch = CNN_pitch.load_from_checkpoint('../models/cnn_pitch.ckpt', lr=lr, num_labels=num_classes, map_location=device)

def predict_instrument(audio_grams, type):
    val = audio_grams.to(device)
    val = val.view(val.size(0), 1, val.size(1), val.size(2))
    val = val.float()
    if type == 'mels':
        prediction = model_mel(val)
    elif type == 'modgds':
        prediction = model_modgd(val)
    else:
        prediction = model_pitch(val)
    soft = nn.Softmax(dim=1)
    prediction = soft(prediction)

    if aggregation_method == "s1":
        prediction = torch.mean(prediction, axis = 0)
    elif aggregation_method == "s2":
        prediction = torch.sum(prediction, axis = 0)
        m = torch.max(prediction)
        prediction /= m
    prediction = prediction.detach().cpu().numpy()
    return prediction

# -----------------------------------------

app = Flask(__name__)

@app.route('/', methods=['GET'])
def render():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    audio_file = request.files['audiofile']

    if audio_file is None or audio_file.filename == "":
        error = 'You have to choose a file to make a prediction.'
        return render_template('index.html', err_empty = error)
    if not allowed_file(audio_file.filename):
        error = 'This file format is not supported. Please choose a .wav file.'
        return render_template('index.html', err_wav = error)
    
    try:
        audio_grams = extract_from_file(audio_file, api = True)

        prediction_mels = predict_instrument(audio_grams['mels'], 'mels')
        prediction_modgds = predict_instrument(audio_grams['modgds'], 'modgds')
        prediction_pitchs = predict_instrument(audio_grams['pitchs'], 'pitchs')

        final_instruments = []
        for i in range(11):
            if prediction_mels[i] > fusion_thresholds[i][0] and prediction_modgds[i] > fusion_thresholds[i][1] and prediction_pitchs[i] > fusion_thresholds[i][2]:
                final_instruments.append(predictions_map[i])

    except:
        error = 'There was an error while predicting, please try again in a few minutes.'
        return render_template('index.html', err_pred = error)
    
    return render_template('index.html', predictions = final_instruments)

@app.route('/predict_test', methods=['POST'])
def predict_test():    
    audio_file = request.files['audiofile']

    if audio_file is None or audio_file.filename == "":
        return jsonify({'error': 'no file'})
    if not allowed_file(audio_file.filename):
        return jsonify({'error': 'format not supported'})

    audio_grams = extract_from_file(audio_file, api = True)

    try:
        prediction_mels = predict_instrument(audio_grams['mels'], 'mels')
        prediction_modgds = predict_instrument(audio_grams['modgds'], 'modgds')
        prediction_pitchs = predict_instrument(audio_grams['pitchs'], 'pitchs')

        dict = {}
        i = 0
        for key in label_map.keys():
            if prediction_mels[i] > fusion_thresholds[i][0] and prediction_modgds[i] > fusion_thresholds[i][1] and prediction_pitchs[i] > fusion_thresholds[i][2]:
                dict[key] = 1
            else:
                dict[key] = 0
            i = i + 1
                
        return dict
    
    except:
        return jsonify({'error': 'error during prediction'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)