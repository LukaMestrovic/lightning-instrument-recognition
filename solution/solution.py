
import os
import requests
import json

test_folder = "../data/test_dataset"

BASE = "http://127.0.0.1:5050/predict_test"


files = [file[:-4] for file in os.listdir(test_folder) if file.endswith('.wav')]

# Sort the files based on the numerical value of N
sorted_files = sorted(files, key=lambda x: int(x.split('_')[1]))

file_dict = {}
for file in sorted_files:
    file_path = os.path.join(test_folder, file)
    file_path = file_path + '.wav'
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            audio_file = f.read()
        filename = os.path.basename(file_path)
        response = requests.post(BASE, files = {"audiofile" : audio_file})    
        file_dict[filename[:-4]] = response.json()

with open("solution.json", "w") as f:
    json.dump(file_dict, f, indent=4, separators=(", ", ": "))
    f.write("\n")

