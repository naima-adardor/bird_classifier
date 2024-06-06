import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import joblib
import numpy as np
import librosa
import plotly.express as px
import wikipedia
import requests


@st.cache_data
def fetch_wikipedia_info(bird_name):
    try:
        page = wikipedia.page(bird_name)
        first_section = next(iter(page.sections), None)
        if first_section:
            title = first_section.title
            text = first_section.text
            return title, text
        else:
            return bird_name, page.summary
    except wikipedia.exceptions.PageError:
        return bird_name, "No information available on Wikipedia."

# Load the model
# model = joblib.load('audio_classifier_model.joblib')

class_mapping_data = pd.read_csv('unique_merged_data.csv')

# @st.cache_data
# def extract_features(file_path):
#     audio, sample_rate = librosa.load(file_path)
#     mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
#     flattened_features = np.mean(mfccs.T, axis=0)
#     return flattened_features

# Set page configuration
st.set_page_config(page_title="Bird Classification", page_icon=":bar_chart:", layout="wide")
# st.sidebar.image("./birdd.gif", use_column_width=True)

st.markdown(
    """
    ## 🐦Bird Call Identifier: Discover the Symphony of Nature!
    Have you ever been enchanted by the symphony of birdsong in the early morning and wondered which birds are serenading you? Our cutting-edge application can help you unlock the secrets of these beautiful melodies.
    **Let's dive in!** 
    """
)

audio_file = st.file_uploader("Upload", type=["ogg", "mp3", "wav"], label_visibility="collapsed")
if audio_file is not None:
    response = requests.post('http://backend:8000/predict/', files={"file": audio_file})
    if response.status_code == 200:
        prediction = response.json()
        st.write(f"The bird is: {prediction['scientific_name']}")

        wiki_title, wiki_info = fetch_wikipedia_info(prediction['scientific_name'])
        st.write(f"**{wiki_title}**")
        st.write(f"{wiki_info}")

        tmp = class_mapping_data[class_mapping_data['scientific_name'] == prediction['scientific_name']].copy()
        fig = px.scatter_mapbox(tmp, lat="latitude", lon="longitude", color="primary_label", zoom=10, title='Bird Recordings Location')
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)