import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import joblib
import numpy as np
import librosa

# Load the model
model = joblib.load('audio_classifier_model.joblib')
class_mapping_data = pd.read_csv('unique_merged_data.csv')  # Adjust the path as necessary


# Define the feature extraction function
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    flattened_features = np.mean(mfccs.T, axis=0)
    return flattened_features

# Set page configuration
st.set_page_config(
    page_title="Bird Classification",
    page_icon=":bar_chart:",
    layout="wide"
)

st.sidebar.image("birdd.gif", use_column_width=True)

sidebar_style = """
    <style>
    [data-testid="stSidebar"] {
        background-color: #EEEE95;  /* Replace this with your desired color */
        background-image: none;  /* Remove any existing background image */

    }
  
    </style>
"""

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    .sidebar-title {
        font-family: 'Reddit Mono', sans-serif;
        text-align: center;
        color: #e59A62; 
        font-weight:bold;
        font-size: 26px;
    }
    .sidebar-text {
        font-size: 16px;
        color: #413C3C;
         font-family: 'Reddit Mono', sans-serif;
         font-size: 16px;
         text-align: center;
    }

    </style>

    """,
    unsafe_allow_html=True
)

# Add text to the sidebar with the custom class
st.sidebar.markdown("""<h1 class="sidebar-title">Guess That Bird!</h1>
                    <p class="sidebar-text" >Start your journey into the captivating world of bird sounds.</p>"""
                    , unsafe_allow_html=True)

# Display GIF using HTML and JavaScript
# Inject JavaScript using a hidden `<div>`
# set it in the sidebar
# Inject CSS with Markdown
st.markdown(sidebar_style, unsafe_allow_html=True)
st.markdown(
    """
    ## 🐦Bird Call Identifier: Discover the Symphony of Nature!

   Have you ever been enchanted by the symphony of birdsong in the early morning and wondered which birds are serenading you? Our cutting-edge application can help you unlock the secrets of these beautiful melodies.

    **Let's dive in!** 
    """
)

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
                padding-top: 0px;
                font-color: #e59A62;
    }






    </style>
"""
# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)
# Add the file uploader with the customized style
audio_file = st.file_uploader("", type=["ogg", "mp3", "wav"])
# Check if audio file is uploaded
if audio_file is not None:
    features = extract_features(audio_file)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)

    for index, row in class_mapping_data.iterrows():
        if row['encoded_label_y'] == prediction[0]:
            st.write(f"The predicted bird species is: {row['primary_label']}")
 

# Add a button

# Add a button
if st.button("Predict Bird Call", key="button1"):
    st.write("Button Clicked!")

# Apply CSS to modify button appearance
button_style = """
    <style>
        div.stButton > button {
            background-color: #e59A62;
            color: black;
            height: 50px;
            width: 200px;
            font-size: 16px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            item-align: center;
        }
        div.stButton > button:hover {
            background-color: #A75211;
            color: white;
        }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)