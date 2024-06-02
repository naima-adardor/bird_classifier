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

def fetch_wikipedia_info(bird_name):
    try:
        page = wikipedia.page(bird_name)
        # Get the first section or big title
        first_section = next(iter(page.sections), None)
        if first_section:
            title = first_section.title
            text = first_section.text
            
             # Take the first image if available
            
            return title, text
        else:
            return bird_name, page.summary  # Fallback to summary if no section is found
    except wikipedia.exceptions.PageError:
        return bird_name, "No information available on Wikipedia."

# Load the model
model = joblib.load('audio_classifier_model.joblib')
class_mapping_data = pd.read_csv('unique_merged_data.csv')  # Adjust the path as necessary


# Define the feature extraction function
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    flattened_features = np.mean(mfccs.T, axis=0)
    return flattened_features


# Function to fetch Wikipedia information
#def fetch_wikipedia_info(bird_name):
    # wiki_wiki = wikipediaapi.Wikipedia('en', user_agent='adardournaima70@gmail.com')

    # page = wiki_wiki.page(bird_name)
    # if page.exists():
    #     # Get the first section or big title
    #     first_section = next(iter(page.sections), None)
    #     if first_section:
    #         return first_section.title, first_section.text
    #     else:
    #         return bird_name, page.summary  # Fallback to summary if no section is found
    # else:
    #     return bird_name, "No information available on Wikipedia."
    #pass


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

    # Find the first occurrence of the predicted encoded_label
    label_info = class_mapping_data[class_mapping_data['encoded_label'] == prediction[0]].iloc[0]
    st.write(f"The bird is: {label_info['scientific_name']}")

# Fetch Wikipedia information and the first image
    wiki_title, wiki_info = fetch_wikipedia_info(label_info['scientific_name'])
    st.write(f"**{wiki_title}**")
    st.write(f"{wiki_info}")

    # Create a DataFrame for plotting
    tmp = class_mapping_data[class_mapping_data['encoded_label'] == prediction[0]].copy()
    
    # Create a scatter mapbox plot
    fig = px.scatter_mapbox(
        tmp,
        lat="latitude",
        lon="longitude",
        color="primary_label",
        zoom=10,
        title='Bird Recordings Location'
    )

    # Update the layout of the plot to use the "open-street-map" style for the map background
    fig.update_layout(mapbox_style="open-street-map")

    # Update the layout of the plot to set the margin around the map
    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

    # Display the scatter mapbox plot
    st.plotly_chart(fig, use_container_width=True)
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