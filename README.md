# lightning-instrument-recognition
Instrument recognition with IRMAS dataset using Pytorch Lightning framework.\
To run the web app, follow these steps:
* Navigate to the project folder lightning-instrument-recognition.
* Create conda environment with 'conda env create -f environment.yml'
* Activate conda environment with 'conda activate lightning-instrument-recognition'
* Navigate to the deploy folder of the project by typing 'cd deploy'.
* Run the python script ‘rest\_api.py’ by typing ‘python rest\_api.py’ which will run the Flask server.
* Open your browser and type 'http://localhost:5050'. This will display the web interface of the app.
* Click on the ‘Upload file’ button and select an audio file with a .wav extension.
* Once the file has been uploaded, click on the ‘Predict’ button and wait for a few seconds until the web page displays all the instruments that appear in the uploaded audio file.\

For more details, see the project and technical documentation.
