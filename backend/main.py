from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import subprocess
import uvicorn
import joblib
import numpy as np
import pandas as pd
import librosa
import wikipedia

app = FastAPI()

# Load the model
model = joblib.load('audio_classifier_model.joblib')
class_mapping_data = pd.read_csv('unique_merged_data.csv')

def extract_features(audio_bytes):
    audio, sample_rate = librosa.load(audio_bytes)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    flattened_features = np.mean(mfccs.T, axis=0)
    return flattened_features

# Define FastAPI endpoints
@app.get("/")
async def read_root():
    return {"message": "Welcome to the model API!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
        features = extract_features(file.filename)
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)

        # Find the first occurrence of the predicted encoded_label
        label_info = class_mapping_data[class_mapping_data['encoded_label'] == prediction[0]].iloc[0]
        return JSONResponse(content={"scientific_name": label_info['scientific_name']})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)