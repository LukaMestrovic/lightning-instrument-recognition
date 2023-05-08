from hydra import compose, initialize
import numpy as np
import pickle
import librosa
import os
import scipy.signal 
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import torch

# global initialization
initialize(version_base=None, config_path="../../configs")
cfg = compose(config_name="config")

# constants
num_classes = cfg.constants.num_classes

# preprocessing
version = cfg.preprocessing.version
sample_rate = cfg.preprocessing.sample_rate
mono = cfg.preprocessing.mono
step_perc = cfg.preprocessing.step_perc
seconds = cfg.preprocessing.seconds
n_fft_mel = cfg.preprocessing.n_fft_mel
hop_length_mel = cfg.preprocessing.hop_length_mel
n_frequency_bins_modgd = cfg.preprocessing.n_frequency_bins_modgd
frame_size_modgd = cfg.preprocessing.frame_size_modgd
hop_size_modgd = cfg.preprocessing.hop_size_modgd
alpha = cfg.preprocessing.alpha
gamma = cfg.preprocessing.gamma
nc = cfg.preprocessing.nc
hop_length_pitch = cfg.preprocessing.hop_length_pitch

# training 
split_perc = cfg.training.split_perc

# paths
train_folder = cfg.paths.train_folder
test_folder = cfg.paths.test_folder

# label_map
label_map = cfg.label_map


def create_melspectrogram(x, sr, n_fft, hop_length):
    
    X = librosa.core.stft(x, hop_length=hop_length, n_fft=n_fft)
    spectrogram = np.abs(X)
    mel_spectrogram = librosa.amplitude_to_db(spectrogram)
    
    return mel_spectrogram

def create_modgdgram(signal, sr, n_frequency_bins=128, frame_size=0.05, hop_size=0.01, alpha=0.9, gamma=0.5, nc=10):
    
    frame_length = int(frame_size * sr)
    hop_length = int(hop_size * sr)
    
    n_time_points = int(len(signal) / sr) * 98
   
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length, axis=0)
    n_frames = len(frames)
   
    #compute modgdgram
    modgdgram = np.zeros((n_frames, frame_length))
    for index, x in enumerate(frames):
        #compute modgdgram for a frame
        n = np.arange(1, len(x) + 1)
        y = n * x
        X = scipy.fft.fft(x)
        XR = np.real(X)
        XI = np.imag(X)

        Y = scipy.fft.fft(y)
        YR = np.real(Y)
        YI = np.imag(Y)

        X_mag = np.abs(X) + 1e-10
        cepstrum = np.fft.ifft(np.log(X_mag)).real #complex cepstrum
        l_w = np.zeros_like(cepstrum)
        l_w[:nc] = 1.0 #low-order cepstral window
        cepstrum_smoothed = cepstrum * l_w
        S = np.abs(np.exp(np.fft.fft(cepstrum_smoothed))) #cepstrally smoothed version of X
        sign = np.sign(np.multiply(XR, YR) + np.multiply(XI, YI))
        module = np.power(np.abs(np.divide(np.multiply(XR, YR) + np.multiply(XI, YI), np.power(S, 2*gamma))), alpha)
        tau = np.multiply(sign, module)

        modgdgram[index] = tau

    modgdgram = np.transpose(modgdgram)
    modgdgram = librosa.amplitude_to_db(modgdgram)
    modgdgram = modgdgram[:n_frequency_bins, :n_time_points]
   
    return modgdgram

def create_pitchgram(x, sr, hop_length):
    
    pitchgram = librosa.feature.chroma_cqt(y=x, sr=sr, hop_length=hop_length, n_chroma=36)
    
    return pitchgram

def extract_from_file(filename, instrument = "pol", api = False):

    y, sr = librosa.load(filename, sr=sample_rate, mono=mono)
    dur = int(len(y) / sr)
    
    melspectrogram = create_melspectrogram(y, sr, n_fft=n_fft_mel, hop_length=hop_length_mel)

    modgdgram = create_modgdgram(y, sr, n_frequency_bins=n_frequency_bins_modgd, frame_size=frame_size_modgd, 
                                 hop_size=hop_size_modgd, alpha=alpha, gamma=gamma, nc=nc)
    
    pitchgram = create_pitchgram(y, sr, hop_length=hop_length_pitch)

    # Segementation of the spectogram
    seg_dur_mel = int(100 * seconds)
    seg_dur_modgd = int(98 * seconds)
    seg_dur_pitch = int(100 * seconds)

    mels = []
    modgds = []
    pitchs = []
    if instrument == "pol":
        for idx in range(0, melspectrogram.shape[1] - seg_dur_mel + 1, int(step_perc * seg_dur_mel)):
            mels.append(melspectrogram[:, idx : (idx + seg_dur_mel)])
        for idx in range(0, modgdgram.shape[1] - seg_dur_modgd + 1, int(step_perc * seg_dur_modgd)):
            modgds.append(modgdgram[:, idx : (idx + seg_dur_modgd)])
        for idx in range(0, pitchgram.shape[1] - seg_dur_pitch + 1, int(step_perc * seg_dur_pitch)):
            pitchs.append(pitchgram[:, idx : (idx + seg_dur_pitch)])   
    else:
        for idx in range(0, melspectrogram.shape[1] - seg_dur_mel + 1, seg_dur_mel):
            mels.append(melspectrogram[:, idx : (idx + seg_dur_mel)])
        for idx in range(0, modgdgram.shape[1] - seg_dur_modgd + 1, seg_dur_modgd):
            modgds.append(modgdgram[:, idx : (idx + seg_dur_modgd)])
        for idx in range(0, pitchgram.shape[1] - seg_dur_pitch + 1, seg_dur_pitch):
            pitchs.append(pitchgram[:, idx : (idx + seg_dur_pitch)])  
    mels = np.array(mels)
    mels = torch.from_numpy(mels)
    modgds = np.array(modgds)
    modgds = torch.from_numpy(modgds)
    pitchs = np.array(pitchs)
    pitchs = torch.from_numpy(pitchs)

    features = {}
    
    features["mels"] = mels
    features["modgds"] = modgds
    features["pitchs"] = pitchs
    if(api == False):
        features["labels"] = torch.zeros([num_classes])
        
        if(instrument == "pol"):
            with open(filename[:-4] + '.txt', 'r') as fp:
                lines = fp.readlines()
                for l in lines:
                    features["labels"][label_map[l[:3]]] = 1
                    
        else:
            features["labels"][label_map[instrument]] = 1
                
    
    return features

def main_training():
    
    features = []
    for instrument in label_map.keys():
        print("Entering folder ", train_folder + instrument)

        for root, dirs, files in os.walk(train_folder + instrument):
            total_files = len(files)

            count = 0
            for file in files:
                if file.endswith('.wav'):
                    count += 1
                    print(count, "/", total_files)
                    feat = extract_from_file(train_folder + instrument +  "/" + file, instrument=instrument)
                    features.append(feat)
                    
    return features

def main_testing():
    
    print("Entering folder ", test_folder)

    features = []
    for root, dirs, files in os.walk(test_folder):
        total_files = int(len(files)/2)

        count = 0
        for file in files:
            if file.endswith('.wav'):
                count += 1
                print(count, "/", total_files)
                feat = extract_from_file(test_folder + file)
                features.append(feat)
    return features

# function definitions
# ----------------------------------------------
# main code

if __name__ == "__main__":
    if "train" in version:
        print("Training data preprocessing started!")
        train_data = main_training()
        print("Training data preprocessing finished!")

        print("Spliting train data!")

        # Convert numpy arrays to PyTorch tensors
        train_data_tensors = [feat["mels"] for feat in train_data]
        X_mel = torch.cat(train_data_tensors, dim=0).unsqueeze(3)

        train_data_tensors = [feat["modgds"] for feat in train_data]
        X_modgd = torch.cat(train_data_tensors, dim=0).unsqueeze(3)

        train_data_tensors = [feat["pitchs"] for feat in train_data]
        X_pitch = torch.cat(train_data_tensors, dim=0).unsqueeze(3)

        y = torch.cat([torch.tensor(feat["labels"]).repeat(len(feat["mels"]), 1) for feat in train_data])

        # Randomly split training data -> 85% training set, 15% validation set
        I = torch.arange(y.shape[0])
        I_train, I_val, _, _ = train_test_split(I, I, test_size=(1 - split_perc), random_state=42)

        X_train_mel = torch.zeros((int(split_perc * len(I)), list(X_mel.size())[1], list(X_mel.size())[2], list(X_mel.size())[3]))
        X_val_mel = torch.zeros((len(I) - len(X_train_mel), list(X_mel.size())[1], list(X_mel.size())[2], list(X_mel.size())[3]))
        X_train_modgd = torch.zeros((int(split_perc * len(I)), list(X_modgd.size())[1], list(X_modgd.size())[2], list(X_modgd.size())[3]))
        X_val_modgd = torch.zeros((len(I) - len(X_train_modgd), list(X_modgd.size())[1], list(X_modgd.size())[2], list(X_modgd.size())[3]))
        X_train_pitch = torch.zeros((int(split_perc * len(I)), list(X_pitch.size())[1], list(X_pitch.size())[2], list(X_pitch.size())[3]))
        X_val_pitch = torch.zeros((len(I) - len(X_train_pitch), list(X_pitch.size())[1], list(X_pitch.size())[2], list(X_pitch.size())[3]))
        y_train = torch.zeros((int(split_perc * len(I)), num_classes))
        y_val = torch.zeros((len(I) - len(y_train), num_classes))

        for i in range(len(I_train)):
            X_train_mel[i] = X_mel[I_train[i]]
            X_train_modgd[i] = X_modgd[I_train[i]]
            X_train_pitch[i] = X_pitch[I_train[i]]
            y_train[i] = y[I_train[i]]
            
        for i in range(len(I_val)):
            X_val_mel[i] = X_mel[I_val[i]]
            X_val_modgd[i] = X_modgd[I_val[i]]
            X_val_pitch[i] = X_pitch[I_val[i]]
            y_val[i] = y[I_val[i]]

        print("Finished splitting!")

        # Save Numpy arrays and dictionaries
        print("Saving training data!")

        X_train_mel = X_train_mel.view(X_train_mel.size(0), 1, X_train_mel.size(1), X_train_mel.size(2))
        X_val_mel = X_val_mel.view(X_val_mel.size(0), 1, X_val_mel.size(1), X_val_mel.size(2))
        X_train_modgd = X_train_modgd.view(X_train_modgd.size(0), 1, X_train_modgd.size(1), X_train_modgd.size(2))
        X_val_modgd = X_val_modgd.view(X_val_modgd.size(0), 1, X_val_modgd.size(1), X_val_modgd.size(2))
        X_train_pitch = X_train_pitch.view(X_train_pitch.size(0), 1, X_train_pitch.size(1), X_train_pitch.size(2))
        X_val_pitch = X_val_pitch.view(X_val_pitch.size(0), 1, X_val_pitch.size(1), X_val_pitch.size(2))

        torch.save(X_train_mel, "../../data/processed/X_train_mel.pt")
        torch.save(X_val_mel, "../../data/processed/X_val_mel.pt")
        torch.save(X_train_modgd, "../../data/processed/X_train_modgd.pt")
        torch.save(X_val_modgd, "../../data/processed/X_val_modgd.pt")
        torch.save(X_train_pitch, "../../data/processed/X_train_pitch.pt")
        torch.save(X_val_pitch, "../../data/processed/X_val_pitch.pt")
        torch.save(y_train, "../../data/processed/y_train.pt")
        torch.save(y_val, "../../data/processed/y_val.pt")

        print("Saved training data!")
    if "test" in version:
        print("Testing started!")
        test_data = main_testing()
        print("Testing finished!")

        print("Test data!")

        # Initialize the testing feature and label dictionaries
        X_test_mel = OrderedDict()
        X_test_modgd = OrderedDict()
        X_test_pitch = OrderedDict()
        y_test = OrderedDict()

        # Store the number of audio fragments per testing file
        num_fragments_per_file = [len(test_data[i]['mels']) for i, _ in enumerate(test_data)]

        mel_shape = [test_data[0]['mels'][0].size(0), test_data[0]['mels'][0].size(1)]
        modgd_shape = [test_data[0]['modgds'][0].size(0), test_data[0]['modgds'][0].size(1)]
        pitch_shape = [test_data[0]['pitchs'][0].size(0), test_data[0]['pitchs'][0].size(1)]

        # Fill the test data dictionaries
        for ix, _ in enumerate(test_data):
            # Initialize the feature and lable matrices for test file at index ix
            X_test_file_ix_mel = torch.zeros((num_fragments_per_file[ix], 1 ,mel_shape[0], mel_shape[1]))
            X_test_file_ix_modgd = torch.zeros((num_fragments_per_file[ix], 1 ,modgd_shape[0], modgd_shape[1]))
            X_test_file_ix_pitch = torch.zeros((num_fragments_per_file[ix], 1 ,pitch_shape[0], pitch_shape[1]))
            y_test_file_ix = torch.zeros((num_fragments_per_file[ix], 1 ,num_classes))

            label = test_data[ix]["labels"]

            j = 0
            for feat in test_data[ix]['mels']:
                X_test_file_ix_mel[j,:,:,:] = feat
                y_test_file_ix[j,:] = label
                j+=1
            
            j = 0
            for feat in test_data[ix]['modgds']:
                X_test_file_ix_modgd[j,:,:,:] = feat
                j+=1

            j = 0
            for feat in test_data[ix]['pitchs']:
                X_test_file_ix_pitch[j,:,:,:] = feat
                j+=1

            X_test_mel[ix] = X_test_file_ix_mel
            X_test_modgd[ix] = X_test_file_ix_modgd
            X_test_pitch[ix] = X_test_file_ix_pitch
            y_test[ix] = y_test_file_ix

        print("Finished test data!")

        print("Saving test data!")

        f = open("../../data/processed/X_test_mel.pkl", "wb")
        pickle.dump(X_test_mel, f)
        f.close()

        f = open("../../data/processed/X_test_modgd.pkl", "wb")
        pickle.dump(X_test_modgd, f)
        f.close()

        f = open("../../data/processed/X_test_pitch.pkl", "wb")
        pickle.dump(X_test_pitch, f)
        f.close()

        f = open("../../data/processed/y_test.pkl", "wb")
        pickle.dump(y_test, f)
        f.close()

        print("Test data saved!")

