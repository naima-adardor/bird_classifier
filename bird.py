import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import librosa
import noisereduce as nr
import plotly.express as px
import wikipedia
from io import BytesIO
import soundfile as sf

# Function to fetch Wikipedia information
@st.cache_data
def fetch_wikipedia_info(bird_name):
    try:
        page = wikipedia.page(bird_name)
        # Get the first section or big title
        first_section = next(iter(page.sections), None)
        if first_section:
            title = first_section.title
            text = first_section.text
            return title, text
        else:
            return bird_name, page.summary  # Fallback to summary if no section is found
    except wikipedia.exceptions.PageError:
        return bird_name, "No information available on Wikipedia."

# Function to normalize and denoise audio
def normalize_and_denoise(audio, sample_rate):
    # Normalize audio data
    audio = audio / np.max(np.abs(audio))
    # Apply Spectral Subtraction Filter
    filtered_audio = nr.reduce_noise(y=audio, sr=sample_rate)
    return filtered_audio

# Function to extract features from audio file
@st.cache_data
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    flattened_features = np.mean(mfccs.T, axis=0)
    return flattened_features, audio, sample_rate

# Set page configuration
st.set_page_config(
    page_title="Bird Classification",
    page_icon=":bar_chart:",
    layout="wide"
)

# Sidebar configuration
st.sidebar.image("birdd.gif", use_column_width=True)
st.sidebar.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #EEEE95;
        background-image: none;
    }
    </style>
    <h1 class="sidebar-title">Guess That Bird!</h1>
    <p class="sidebar-text">Start your journey into the captivating world of bird sounds.</p>
""", unsafe_allow_html=True)

# Main content
st.markdown("""
    ## 🐦Bird Call Identifier: Discover the Symphony of Nature!
    Have you ever been enchanted by the symphony of birdsong in the early morning and wondered which birds are serenading you? Our cutting-edge application can help you unlock the secrets of these beautiful melodies.
    **Let's dive in!**
""")

# Custom CSS for file uploader
custom_css = """
<style>
    [data-testid='stFileUploader'] {
        align-items: center;
        background-color: #FFFFFF;
        border-radius: 5px;
        border: 2px dashed #e59A62;
        align-self: center;
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 10px;
        cursor: pointer;
        padding: 20px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Add the file uploader with the customized style
audio_file = st.file_uploader("Upload an audio file (ogg, mp3, wav)", type=["ogg", "mp3", "wav"], label_visibility="collapsed")

# Check if audio file is uploaded
if audio_file is not None:
    # Extract features from the uploaded audio file
    features, audio, sample_rate = extract_features(audio_file)
    features = np.array(features).reshape(1, -1)
    
    # Normalize and denoise the audio
    filtered_audio = normalize_and_denoise(audio, sample_rate)
    
    # Play the original and filtered audio files
    st.text("Playing original uploaded audio...")
    st.audio(audio_file)

    # Convert filtered audio to WAV format for playback
    audio_buffer = BytesIO()
    sf.write(audio_buffer, filtered_audio, sample_rate, format='wav')
    audio_buffer.seek(0)
    st.text("Playing filtered audio...")
    st.audio(audio_buffer)

    # Load the trained model and class mapping data
    model = joblib.load('audio_classifier_model.joblib')
    class_mapping_data = pd.read_csv('unique_merged_data.csv')  # Adjust the path as necessary

    # Predict the bird species
    prediction = model.predict(features)
    label_info = class_mapping_data[class_mapping_data['encoded_label'] == prediction[0]].iloc[0]

    # Display the predicted bird species
    st.write(f"The predicted bird species is: {label_info['scientific_name']}")

    # Fetch Wikipedia information for the predicted bird species
    wiki_title, wiki_info = fetch_wikipedia_info(label_info['scientific_name'])
    st.subheader(f"Wikipedia Information: {wiki_title}")
    st.write(wiki_info)

    # Filter the data to include only the recordings of the predicted bird species
    filtered_data = class_mapping_data[class_mapping_data['encoded_label'] == prediction[0]]

    # Define the boundaries for the Western Ghats region
    lower_latitude = 8
    upper_latitude = 22
    lower_longitude = 73
    upper_longitude = 79

    # Further filter the data to include only recordings within the Western Ghats region
    filtered_data = filtered_data[(filtered_data['latitude'] >= lower_latitude) & 
                                  (filtered_data['latitude'] <= upper_latitude) &
                                  (filtered_data['longitude'] >= lower_longitude) &
                                  (filtered_data['longitude'] <= upper_longitude)]

    # Create a scatter mapbox plot for the filtered data
    fig = px.scatter_mapbox(
        filtered_data,
        lat="latitude",
        lon="longitude",
        color="scientific_name",
        zoom=5,
        center={"lat": (lower_latitude + upper_latitude) / 2, "lon": (lower_longitude + upper_longitude) / 2},
        title="Bird Recordings Location in the Western Ghats"
    )
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0, "t":30, "l":0, "b":0})
    st.plotly_chart(fig, use_container_width=True)
