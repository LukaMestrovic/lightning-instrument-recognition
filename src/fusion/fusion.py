from hydra import compose, initialize
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning.pytorch as L
import pickle
import sys
import os
sys.path.append("../")
from models.lightning_modules import CNN_mel, CNN_modgd, CNN_pitch, ListDataset


# global initialization
initialize(version_base=None, config_path="../../configs")
cfg = compose(config_name="config")

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_acc(y1, y2):
    ret = 0
    for i, _ in enumerate(y1):
        ret += int(y1[i] == y2[i])
    ret /= len(y1)
    return ret

def calculate_f1(y1, y2):
    tp = 0
    fp = 0
    fn = 0
    for i, _ in enumerate(y1):
        if y1[i] == 1 and y2[i] == 1:
            tp += 1
        elif y1[i] == 1 and y2[i] == 0:
            fp += 1
        elif y1[i] == 0 and y2[i] == 1:
            fn += 1
    f1 = (2 * tp) / (2 * tp + fp + fn)
    return f1

# constants
lr = cfg.training.learning_rate
num_classes = cfg.constants.num_classes
num_workers = os.cpu_count()

# paths
models_dir = cfg.paths.models_folder

# label map
label_map = cfg.label_map

# methods
aggregation_method = cfg.aggregation.method
fusion_method = cfg.fusion.method

# fusion
theta_step = cfg.fusion.theta_step
theta_step_new = cfg.fusion.theta_step_new
theta_max = cfg.fusion.theta_max
version = cfg.fusion.version
optimization_metric = cfg.fusion.optimization_metric

models = []
model_names = []

for root, dirs, files in os.walk(models_dir):
    files.sort()
    for file in files:
        if (file == 'cnn_mel.ckpt'):
            print('success')
            model = CNN_mel.load_from_checkpoint(os.path.join(models_dir, file), lr=lr, num_labels=num_classes, map_location=device)
            model_names.append(file)
            models.append(model)
        elif (file == 'cnn_modgd.ckpt'):
            model = CNN_modgd.load_from_checkpoint(os.path.join(models_dir, file), lr=lr, num_labels=num_classes, map_location=device)
            model_names.append(file)
            models.append(model)
        elif (file == 'cnn_pitch.ckpt'):
            model = CNN_pitch.load_from_checkpoint(os.path.join(models_dir, file), lr=lr, num_labels=num_classes, map_location=device)
            model_names.append(file)
            models.append(model)
num_models = len(models)
print(model_names)


print("Load X_test and y_test")
f = open("../../data/processed/X_test_mel.pkl", "rb")
X_test_mel = pickle.load(f)
f.close()
print("Mel data loaded!")

f = open("../../data/processed/X_test_modgd.pkl", "rb")
X_test_modgd = pickle.load(f)
f.close()
print("Modgd data loaded!")

f = open("../../data/processed/X_test_pitch.pkl", "rb")
X_test_pitch = pickle.load(f)
f.close()
print("Pitch data loaded!")

f = open("../../data/processed/y_test.pkl", "rb")
y_test = pickle.load(f)
f.close()

X_test = [X_test_mel, X_test_modgd, X_test_pitch]

a = y_test
y_test = torch.zeros((len(a), num_classes))
for i in range(len(a)):
    y_test[i] = a[i][0]

print("Loaded!")

print("Computing probabilities")
probabilities = []

for i in range(num_models):
    keys = list(X_test[i].keys())
    vals = list(X_test[i].values())
    dataset = ListDataset(keys, vals)
    predict_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=False)

    model = models[i]
    trainer = L.Trainer()
    trainer.predict(model, dataloaders=predict_loader)

    model_predictions = model.model_predictions
    model_predictions = torch.stack(model_predictions)
    probabilities.append(model_predictions)

probabilities = torch.stack(probabilities)
probabilities = probabilities.detach().cpu().numpy()

print("Probabilities computed!")
print(probabilities.shape)
y_pred = np.zeros((probabilities.shape[1], probabilities.shape[2]))
mul_probabilities = np.zeros((probabilities.shape[1], probabilities.shape[2]))
fusion = np.zeros((num_classes, num_models))
thetas = np.arange(0, theta_max, theta_step)

final_accuracy = 0.0

print("Calculating fusion thresholds!")

if fusion_method == 1:
    
    thetas_mel = thetas
    if "mel" not in version:
        thetas_mel = [-1]
    thetas_modgd = thetas
    if "modgd" not in version:
        thetas_modgd = [-1]
    thetas_pitch = thetas
    if "pitch" not in version:
        thetas_pitch = [-1]

    for k in range(num_classes):
        best_acc = 0.0
        best_f1 = 0.0
        for theta1 in thetas_mel:
            print(k, theta1)
            for theta2 in thetas_modgd:
                for theta3 in thetas_pitch:
                    tmp_pred1 = np.array(probabilities[0, :, k]) - theta1
                    tmp_pred2 = np.array(probabilities[1, :, k]) - theta2
                    tmp_pred3 = np.array(probabilities[2, :, k]) - theta3

                    tmp_pred1[tmp_pred1 > 0] = 1
                    tmp_pred1[tmp_pred1 <= 0] = 0
                    tmp_pred2[tmp_pred2 > 0] = 1
                    tmp_pred2[tmp_pred2 <= 0] = 0
                    tmp_pred3[tmp_pred3 > 0] = 1
                    tmp_pred3[tmp_pred3 <= 0] = 0

                    fusion_thetas = []
                    if theta1 == -1:
                        tmp_pred1 = np.ones(tmp_pred1.shape)
                    if theta2 == -1:
                        tmp_pred2 = np.ones(tmp_pred2.shape)
                    if theta3 == -1:
                        tmp_pred3 = np.ones(tmp_pred3.shape)

                    fusion_thetas.append(theta1)
                    fusion_thetas.append(theta2)
                    fusion_thetas.append(theta3)

                    tmp_pred = np.multiply(np.multiply(tmp_pred1, tmp_pred2), tmp_pred3)

                    if optimization_metric == "acc":
                        acc = calculate_acc(tmp_pred, y_test[:, k])
                        if acc > best_acc:
                            best_acc = acc
                            y_pred[:, k] = tmp_pred
                            fusion[k, :] = fusion_thetas
                    
                    elif optimization_metric == "f1":
                        f1 = calculate_f1(tmp_pred, y_test[:, k])
                        if f1 > best_f1:
                            best_f1 = f1
                            y_pred[:, k] = tmp_pred
                            fusion[k, :] = fusion_thetas

        thetas_mel_new = np.arange(fusion[k][0] - 0.05, fusion[k][0] + 0.05, theta_step_new)
        if "mel" not in version:
            thetas_mel_new = [-1]
        thetas_modgd_new = np.arange(fusion[k][1] - 0.05, fusion[k][1] + 0.05, theta_step_new)
        if "modgd" not in version:
            thetas_modgd_new = [-1]
        thetas_pitch_new = np.arange(fusion[k][2] - 0.05, fusion[k][2] + 0.05, theta_step_new)
        if "pitch" not in version:
            thetas_pitch_new = [-1]

        for theta1 in thetas_mel_new:
            print(k, theta1)
            for theta2 in thetas_modgd_new:
                for theta3 in thetas_pitch_new:
                    tmp_pred1 = np.array(probabilities[0, :, k]) - theta1
                    tmp_pred2 = np.array(probabilities[1, :, k]) - theta2
                    tmp_pred3 = np.array(probabilities[2, :, k]) - theta3

                    tmp_pred1[tmp_pred1 > 0] = 1
                    tmp_pred1[tmp_pred1 <= 0] = 0
                    tmp_pred2[tmp_pred2 > 0] = 1
                    tmp_pred2[tmp_pred2 <= 0] = 0
                    tmp_pred3[tmp_pred3 > 0] = 1
                    tmp_pred3[tmp_pred3 <= 0] = 0

                    fusion_thetas = []
                    if theta1 == -1:
                        tmp_pred1 = np.ones(tmp_pred1.shape)
                    if theta2 == -1:
                        tmp_pred2 = np.ones(tmp_pred2.shape)
                    if theta3 == -1:
                        tmp_pred3 = np.ones(tmp_pred3.shape)

                    fusion_thetas.append(theta1)
                    fusion_thetas.append(theta2)
                    fusion_thetas.append(theta3)

                    tmp_pred = np.multiply(np.multiply(tmp_pred1, tmp_pred2), tmp_pred3)

                    if optimization_metric == "acc":
                        acc = calculate_acc(tmp_pred, y_test[:, k])
                        if acc > best_acc:
                            best_acc = acc
                            y_pred[:, k] = tmp_pred
                            fusion[k, :] = fusion_thetas
                    
                    elif optimization_metric == "f1":
                        f1 = calculate_f1(tmp_pred, y_test[:, k])
                        if f1 > best_f1:
                            best_f1 = f1
                            y_pred[:, k] = tmp_pred
                            fusion[k, :] = fusion_thetas

        final_accuracy += best_acc

        print(fusion[k][0], ", ", fusion[k][1], ", ", fusion[k][2], sep = '')

final_accuracy = final_accuracy / 11.0

print("Final accuracy:", final_accuracy)

# Save fusion thresholds
save_str = "fusion_thresholds_"
for model_name in version:
    save_str += model_name
    save_str += "_"
save_str += optimization_metric
save_str += ".npy"
np.save(save_str, fusion)
