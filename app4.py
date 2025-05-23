
import streamlit as st
import pretty_midi
import os
import time
import base64
import numpy as np
import tensorflow as tf
import pickle
from datetime import datetime
import requests
import gdown
from streamlit_lottie import st_lottie
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="NSynth Melody Studio", page_icon="🎹", layout="wide")
# Constants
MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.pkl"
OUTPUT_PATH = "custom_melody.mid"

# Google Drive file ID for model (replace with your actual file ID)
MODEL_DRIVE_ID = "1fIsD4qBUVmxM3QvS0b505BxoyTmc7On7"  # Replace with your model1.h5 file ID

@st.cache_resource
def download_model_from_drive():
    """Download H5 model from Google Drive if it doesn't exist locally."""
    
    if not os.path.exists(MODEL_PATH):
        st.info(f"📥 Downloading H5 model ({MODEL_PATH})... This may take a few minutes for large models.")
        
        try:
            download_url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
            gdown.download(download_url, MODEL_PATH, fuzzy=True, quiet=False)
            st.success(f"✅ H5 model downloaded successfully!")
            return True
            
        except Exception as e:
            st.error(f"❌ Error downloading H5 model: {e}")
            st.info("💡 Make sure your Google Drive link is set to 'Anyone with the link can view'")
            return False
    else:
        st.success(f"✅ H5 model already exists locally!")
        return True

# Import the NSynthGenerator class from the provided code
class NSynthGenerator:
    def __init__(self, model_path: str, scaler_path: str, seq_length: int = 32):
        # Download model from Google Drive if needed
        download_model_from_drive()
        
        self.model = tf.keras.models.load_model(model_path)
        self.model.compile(optimizer="adam", loss="mse")
        self.seq_length = seq_length
        self.scaler = self._load_scaler(scaler_path)

    def _load_scaler(self, scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
            if isinstance(scaler_data, dict) and 'scaler' in scaler_data:
                return scaler_data['scaler']
            else:
                raise TypeError(f"Expected dictionary with 'scaler', got {type(scaler_data)}")

    def create_seed_sequence(self, pitch=60, velocity=100, instrument_family=0, instrument_source=0, random_init=True):
        params = np.array([
            pitch / 127.0,
            velocity / 127.0,
            instrument_family / 11.0,
            instrument_source / 3.0
        ])
        audio_features = np.random.normal(0, 1, 148) if random_init else np.zeros(148)
        initial_vector = np.concatenate([params, audio_features])
        initial_vector = self.scaler.transform(initial_vector.reshape(1, -1))
        seed_sequence = np.tile(initial_vector, (self.seq_length, 1))
        return seed_sequence

    def generate_sequence(self, seed_sequence, num_steps=1000, noise_level=0.01):
        generated_sequence = seed_sequence.copy()
        for _ in range(num_steps):
            input_sequence = np.expand_dims(generated_sequence[-self.seq_length:], axis=0)
            next_vector = self.model.predict(input_sequence, verbose=0)
            noise = np.random.normal(0, noise_level, next_vector.shape)
            next_vector += noise
            next_vector = np.clip(next_vector, -1.5, 1.5)
            generated_sequence = np.vstack([generated_sequence, next_vector])
        return generated_sequence

#
# Custom CSS - Simplified
st.markdown("""
<style>
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 5%;
    padding-right: 5%;
}

/* Background gradient */
.stApp {
    background: linear-gradient(to bottom, #f3f4f6, #dbeafe, #e0f2fe);
}

/* Improvements for header text */
h1, h2, h3 {
    color: #1e40af;
    font-weight: 800;
}

h1 {
    font-size: 2.5rem;
    margin-bottom: 1.5rem;
    padding: 1rem 2rem; /* Adding padding for the box effect */
    border-radius: 10px; /* Rounded corners */
    background: linear-gradient(135deg, #60a5fa, #3b82f6); /* Gradient background */
    color: white; /* White text color for contrast */
    text-align: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); /* Slight shadow to elevate the box */
    display: inline-block; /* Makes the background fit the text */
    border-bottom: 3px solid #3b82f6; /* Border at the bottom */
    transition: all 0.3s ease; /* Smooth transition for hover effect */
}

/* Hover effect for gradient box */
h1:hover {
    background: linear-gradient(135deg, #3b82f6, #60a5fa); /* Reversed gradient for hover */
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.15); /* Stronger shadow on hover */
    transform: scale(1.05); /* Slight scale-up effect */
}


h2 {
    font-size: 1.8rem;
    padding-top: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #93c5fd;
}

h3 {
    font-size: 1.3rem;
    padding-top: 0.8rem;
}

/* Markdown text */
p, li {
    font-size: 1rem;
    line-height: 1.6;
    color: #1f2937;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    x
    border-right: 1px solid #e2e8f0;
    padding: 2rem 1rem;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2 {
    color: #1e40af;
    font-size: 1.5rem;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}
.stButton > button, .stDownloadButton > button, .stButton > button * {
    color: white !important;
}

/* Button styling */
button, .stButton > button, .stDownloadButton > button {
    background: linear-gradient(90deg, #3b82f6, #1d4ed8); /* Blue gradient */
    color: white;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    border: none !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
}

/* Hover Effect */
button:hover, .stButton > button:hover, .stDownloadButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #3b82f6); /* Reversed gradient on hover */
    box-shadow: 0 6px 12px -1px rgba(0, 0, 0, 0.2), 0 4px 8px -1px rgba(0, 0, 0, 0.1) !important;
    color:white;
}

/* Active Effect */
button:active, .stButton > button:active, .stDownloadButton > button:active {
    transform: scale(0.95);
}


button:hover, .stButton>button:hover, .stDownloadButton>button:hover {
    background-color: #2563eb !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
    transform: translateY(-2px);
}

button:active, .stButton>button:active, .stDownloadButton>button:active {
    transform: translateY(0px);
}

/* Card-like effect for sections */
div.row-widget.stRadio > div {
    background-color: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}



i

/* Label styling */
label {
    font-weight: 600 !important;
    color: #4b5563 !important;
    margin-bottom: 0.5rem !important;
}

/* Radio buttons */
.stRadio > div {
    margin-top: 0.5rem;
}

.stRadio label {
    font-weight: 400 !important;
    color: #374151 !important;
}

/* Selectbox styling */
.stSelectbox > div > div {
    background-color: #f9fafb !important;
    border-radius: 8px !important;
    border: 1px solid #d1d5db !important;
}

.stSelectbox > div > div:hover {
    border-color: #3b82f6 !important;
}


/* Expander styling */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    color: #1e40af !important;
    background-color: #eff6ff !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}

.streamlit-expanderHeader:hover {
    background-color: #dbeafe !important;
}

.streamlit-expanderContent {
    background-color: white !important;
    border-radius: 0 0 8px 8px !important;
    padding: 1.5rem !important;
    border: 1px solid #dbeafe !important;
    border-top: none !important;
}

/* Metrics styling */
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #1e40af !important;
}

[data-testid="stMetricLabel"] {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #4b5563 !important;
}

/* Data display elements */
[data-testid="stTable"], .dataframe {
    width: 100%;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
}

[data-testid="stTable"] th, .dataframe th {
    background-color: #eff6ff !important;
    color: #1e40af !important;
    font-weight: 600 !important;
    text-align: left !important;
    padding: 1rem !important;
    border-bottom: 2px solid #dbeafe !important;
}

[data-testid="stTable"] td, .dataframe td {
    background-color: white !important;
    color: #1f2937 !important;
    padding: 0.75rem 1rem !important;
    border-bottom: 1px solid #e5e7eb !important;
}

[data-testid="stTable"] tr:last-child td, .dataframe tr:last-child td {
    border-bottom: none !important;
}

[data-testid="stTable"] tr:hover td, .dataframe tr:hover td {
    background-color: #f9fafb !important;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    height: auto !important;
    padding: 0.75rem 1rem !important;
    background-color: #f3f4f6 !important;
    border-radius: 8px 8px 0 0 !important;
    border: 1px solid #e5e7eb !important;
    border-bottom: none !important;
    color: #4b5563 !important;
    font-weight: 600 !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: #eff6ff !important;
    color: #1e40af !important;
}

.stTabs [aria-selected="true"] {
    background-color: #dbeafe !important;
    color: #1e40af !important;
}

.stTabs [data-baseweb="tab-panel"] {
    background-color: white !important;
    border-radius: 0 0 8px 8px !important;
    border: 1px solid #e5e7eb !important;
    padding: 1.5rem !important;
}

/* Checkbox styling */
.stCheckbox > label > div[role="checkbox"] {
    background-color: #f3f4f6 !important;
    border-color: #d1d5db !important;
}

.stCheckbox > label > div[role="checkbox"] > svg {
    color: #3b82f6 !important;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background-color: #3b82f6 !important;
}

/* Success/Info/Warning/Error messages */
.element-container .stAlert {
    padding: 1rem 1.5rem !important;
    border-radius: 8px !important;
    margin: 1rem 0 !important;
}

/* Success message */
.element-container [data-baseweb="notification"] [data-emotion="css-1knrpii"] {
    background-color: #ecfdf5 !important;
    border-left-color: #10b981 !important;
}

/* Info message */
.element-container [data-baseweb="notification"] [data-emotion="css-1vly3lj"] {
    background-color: #eff6ff !important;
    border-left-color: #3b82f6 !important;
}

/* Warning message */
.element-container [data-baseweb="notification"] [data-emotion="css-17c6prg"] {
    background-color: #fffbeb !important;
    border-left-color: #f59e0b !important;
}

/* Error message */
.element-container [data-baseweb="notification"] [data-emotion="css-g4us6j"] {
    background-color: #fef2f2 !important;
    border-left-color: #ef4444 !important;
}

/* File uploader */
.stFileUploader > div > button {
    background-color: #3b82f6 !important;
    color: white !important;
}

.stFileUploader > div > button:hover {
    background-color: #2563eb !important;
}


/* Date input */
.stDateInput > div > div > input {
    border-radius: 8px !important;
    border: 1px solid #d1d5db !important;
    padding: 0.5rem 1rem !important;
}

.stDateInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3) !important;
}

/* Time input */
.stTimeInput > div > div > input {
    border-radius: 8px !important;
    border: 1px solid #d1d5db !important;
    padding: 0.5rem 1rem !important;
}

.stTimeInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3) !important;
}

/* Container styling for cards */
[data-testid="stVerticalBlock"] {
    gap: 1.5rem;
}

/* Make dividers nicer */
hr {
    border: 0;
    height: 1px;
    background-image: linear-gradient(to right, rgba(59, 130, 246, 0), rgba(59, 130, 246, 0.75), rgba(59, 130, 246, 0));
    margin: 2rem 0;
}

/* Links styling */
a {
    color: #2563eb !important;
    font-weight: 500 !important;
    text-decoration: none !important;
    transition: all 0.2s ease !important;
}

a:hover {
    color: #1d4ed8 !important;
    text-decoration: underline !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: #f1f5f9;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: #94a3b8;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #64748b;
}

/* Code blocks */
code {
    background-color: #f1f5f9 !important;
    padding: 0.2rem 0.4rem !important;
    border-radius: 4px !important;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace !important;
    font-size: 0.9rem !important;
    color: #1e40af !important;
}



/* Media query for responsive design */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    h1 {
        font-size: 2rem;
    }
    
    h2 {
        font-size: 1.5rem;
    }
    
    h3 {
        font-size: 1.2rem;
    }
}

/* Beautiful card class you can apply to containers */
.css-card {
    border-radius: 10px;
    background-color: white;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    padding: 2rem;
    margin-bottom: 1rem;
}


.pulse {
    animation: pulse 2s infinite;
}

/* Nice quote styling */
blockquote {
    border-left: 4px solid #3b82f6;
    padding-left: 1rem;
    font-style: italic;
    color: #4b5563;
    margin: 1.5rem 0;
}
""", unsafe_allow_html=True)

# Instrument options
family_names = {
    0: "Bass", 1: "Piano", 2: "Flute", 3: "Guitar", 4: "Organ", 
    5: "Synth Pad", 6: "Strings", 7: "Brass", 8: "Ensemble", 
    9: "Synth Lead", 10: "Percussion", 11: "Others"
}

family_to_program = {
    0: 32, 1: 0, 2: 73, 3: 24, 4: 16, 5: 88, 6: 40, 7: 61, 8: 48, 
    9: 81, 10: 118, 11: 0
}

# Melody pattern presets
melody_patterns = {
    "Arpeggio": [0, 4, 7, 12],
    "Up-Down": [0, 2, 4, 7, 4, 2],
    "Scale": [0, 2, 4, 5, 7, 9, 11, 12],
    "Simple Loop": [0, 3, 5, 7],
    "Custom": []
}

# Initialize session state values
def init_session_state():
    if 'instrument_tracks' not in st.session_state:
        st.session_state['instrument_tracks'] = [{
            'id': 0,
            'family': 'Piano',
            'phases': [{
                'id': 0,
                'start_time': 0,
                'end_time': 15,
                'base_pitch': 60,
                'velocity': 100,
                'melody_type': 'Arpeggio',
                'pattern': melody_patterns['Arpeggio'].copy(),
                'noise_level': 0.01,
                'custom_pattern': '0,4,7,12'
            }]
        }]
    
    if 'track_counter' not in st.session_state:
        st.session_state['track_counter'] = 1
        
    if 'phase_counter' not in st.session_state:
        st.session_state['phase_counter'] = {0: 1}

# Helper functions
def add_phase(track_id, start_time=0, end_time=15):
    if track_id not in st.session_state['phase_counter']:
        st.session_state['phase_counter'][track_id] = 0
    
    phase_id = st.session_state['phase_counter'][track_id]
    st.session_state['phase_counter'][track_id] += 1
    
    for track in st.session_state['instrument_tracks']:
        if track['id'] == track_id:
            track['phases'].append({
                'id': phase_id,
                'start_time': start_time,
                'end_time': end_time,
                'base_pitch': 60,
                'velocity': 100,
                'melody_type': 'Arpeggio',
                'pattern': melody_patterns['Arpeggio'].copy(),
                'noise_level': 0.01,
                'custom_pattern': '0,4,7,12'
            })
            break

def remove_phase(track_id, phase_id):
    for track in st.session_state['instrument_tracks']:
        if track['id'] == track_id and len(track['phases']) > 1:
            track['phases'] = [phase for phase in track['phases'] if phase['id'] != phase_id]
            break

def add_track(duration=30):
    new_id = st.session_state['track_counter']
    st.session_state['track_counter'] += 1
    st.session_state['phase_counter'][new_id] = 1
    st.session_state['instrument_tracks'].append({
        'id': new_id,
        'family': 'Piano',
        'phases': [{
            'id': 0,
            'start_time': 0,
            'end_time': duration,
            'base_pitch': 60,
            'velocity': 100,
            'melody_type': 'Arpeggio',
            'pattern': melody_patterns['Arpeggio'].copy(),
            'noise_level': 0.01,
            'custom_pattern': '0,4,7,12'
        }]
    })

def remove_track(track_id):
    st.session_state['instrument_tracks'] = [track for track in st.session_state['instrument_tracks'] 
                                         if track['id'] != track_id]

# Compose tab content
def compose_tab():
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## 🎵 Melody Properties")
        
        col1, col2 = st.columns(2)
        with col1:
            duration = st.slider("Duration (seconds)", 5, 120, 30)
        with col2:
            tempo = st.slider("Tempo (BPM)", 60, 180, 120)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🎸 Instruments & Phases")
    
    if st.session_state['instrument_tracks']:
        instrument_tabs = st.tabs([f"{track['family']} Track" for track in st.session_state['instrument_tracks']])
        
        for i, (tab, track) in enumerate(zip(instrument_tabs, st.session_state['instrument_tracks'])):
            with tab:
                track_id = track['id']
                family_options = list(family_names.values())
                family_idx = family_options.index(track['family'])
                track['family'] = st.selectbox("Instrument Type", family_options, 
                                            index=family_idx, key=f"family_{track_id}")
                
                st.markdown("<h3>Phases</h3>", unsafe_allow_html=True)
                
                for phase_idx, phase in enumerate(sorted(track['phases'], key=lambda x: x['start_time'])):
                    phase_id = phase['id']
                    
                    with st.expander(f"Phase {phase_idx+1}: {phase['start_time']}s - {phase['end_time']}s", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            phase['start_time'] = st.slider("Start Time (s)", 0, duration-1, 
                                                         phase['start_time'], key=f"start_{track_id}_{phase_id}")
                            phase['end_time'] = st.slider("End Time (s)", phase['start_time']+1, duration, 
                                                       phase['end_time'], key=f"end_{track_id}_{phase_id}")
                            phase['base_pitch'] = st.slider("Base Pitch", 40, 80, phase['base_pitch'], 
                                                         key=f"pitch_{track_id}_{phase_id}")
                        
                        with col2:
                            phase['velocity'] = st.slider("Velocity", 50, 127, phase['velocity'], 
                                                       key=f"velocity_{track_id}_{phase_id}")
                            phase['melody_type'] = st.selectbox("Melody Pattern", list(melody_patterns.keys()), 
                                                              index=list(melody_patterns.keys()).index(phase['melody_type']),
                                                              key=f"melody_type_{track_id}_{phase_id}")
                            
                            if phase['melody_type'] == "Custom":
                                custom_pattern = st.text_input("Custom Pattern (comma-separated)", 
                                                           phase['custom_pattern'],
                                                           key=f"custom_pattern_{track_id}_{phase_id}")
                                phase['custom_pattern'] = custom_pattern
                                try:
                                    phase['pattern'] = [int(x.strip()) for x in custom_pattern.split(",")]
                                except:
                                    st.warning("Enter valid comma-separated numbers")
                                    phase['pattern'] = [0, 4, 7, 12]
                            else:
                                phase['pattern'] = melody_patterns[phase['melody_type']]
                            
                            phase['noise_level'] = st.slider("Creativity", 0.0, 0.05, phase['noise_level'], 
                                                          step=0.005, key=f"noise_{track_id}_{phase_id}")
                        
                        if len(track['phases']) > 1:
                            if st.button("Remove Phase", key=f"remove_phase_{track_id}_{phase_id}"):
                                remove_phase(track_id, phase_id)
                                st.rerun()
                
                if st.button("+ Add Phase", key=f"add_phase_{track_id}"):
                    if track['phases']:
                        last_phase_end = max([p['end_time'] for p in track['phases']])
                        new_phase_start = last_phase_end
                        new_phase_end = min(new_phase_start + 15, duration)
                    else:
                        new_phase_start = 0
                        new_phase_end = duration
                    
                    add_phase(track_id, new_phase_start, new_phase_end)
                    st.rerun()
                
                if len(st.session_state['instrument_tracks']) > 1:
                    if st.button("- Remove Instrument Track", key=f"remove_track_{track_id}"):
                        remove_track(track_id)
                        st.rerun()
    
    if len(st.session_state['instrument_tracks']) < 6:
        if st.button("+ Add Instrument Track"):
            add_track(duration)
            st.rerun()
    else:
        st.info("Maximum of 6 instrument tracks reached")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎵 Generate Melody", use_container_width=True):
            generate_melody(duration, tempo)

# Settings tab content
def settings_tab():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## ⚙️ Advanced Settings")
    
    with st.form("advanced_settings"):
        st.write("These settings will apply to newly created instruments and phases.")
        
        col1, col2 = st.columns(2)
        with col1:
            default_noise = st.slider("Default Creativity Level", 0.0, 0.05, 0.01, step=0.001)
            default_velocity = st.slider("Default Velocity", 50, 127, 100)
        
        with col2:
            default_pattern = st.selectbox("Default Melody Pattern", list(melody_patterns.keys()))
            default_pitch = st.slider("Default Base Pitch", 40, 80, 60)
        
        if st.form_submit_button("Save Default Settings"):
            st.session_state['default_noise'] = default_noise
            st.session_state['default_velocity'] = default_velocity
            st.session_state['default_pattern'] = default_pattern
            st.session_state['default_pitch'] = default_pitch
            st.success("Default settings saved!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Melody generation function
def generate_melody(duration, tempo):
    try:
        progress_placeholder = st.empty()
        progress_placeholder.markdown('<div class="info-message">🎼 Generating your melody...</div>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        time_step = 60 / tempo / 4
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        
        for track_idx, track in enumerate(st.session_state['instrument_tracks']):
            instrument = pretty_midi.Instrument(
                program=family_to_program[list(family_names.values()).index(track['family'])],
                name=f"Track {track_idx+1}: {track['family']}"
            )
            
            sorted_phases = sorted(track['phases'], key=lambda x: x['start_time'])
            
            for phase in sorted_phases:
                phase_duration = phase['end_time'] - phase['start_time']
                phase_steps = int(phase_duration / time_step * 4)
                
                generator = NSynthGenerator(MODEL_PATH, SCALER_PATH)
                
                seed = generator.create_seed_sequence(
                    pitch=phase['base_pitch'],
                    velocity=phase['velocity'],
                    instrument_family=list(family_names.values()).index(track['family']),
                    instrument_source=0,
                    random_init=True
                )
                
                generated = generator.generate_sequence(seed, num_steps=phase_steps, noise_level=phase['noise_level'])
                
                pattern = phase['pattern']
                for i, vec in enumerate(generated):
                    jitter = np.random.uniform(-0.02, 0.02)
                    note_time = phase['start_time'] + (i * time_step) + jitter
                    
                    if note_time < phase['start_time'] or note_time >= phase['end_time']:
                        continue
                    
                    pattern_position = (i // 4) % len(pattern)
                    
                    raw_pitch = np.clip(round(vec[0] * 127), 0, 127)
                    note_velocity = int(np.clip(round(vec[1] * 127), 30, 127))
                    
                    pattern_strength = 0.8 if i % 8 == 0 else 0.5
                    pattern_pitch = phase['base_pitch'] + pattern[pattern_position]
                    note_pitch = int(round(pattern_pitch * pattern_strength + raw_pitch * (1 - pattern_strength)))
                    note_pitch = np.clip(note_pitch, 0, 127)
                    
                    if i % 8 < 4:
                        note_velocity = int(note_velocity * np.random.uniform(0.8, 1.0))
                    else:
                        note_velocity = int(note_velocity * np.random.uniform(0.5, 0.8))
                    note_velocity = np.clip(note_velocity, 30, 127)
                    
                    if note_velocity > 20:
                        note_duration = time_step * (1.2 + (0.8 * (i % 4 == 0)))
                        note_end = min(note_time + note_duration, phase['end_time'] - 0.01)
                        
                        if note_end > note_time + 0.05:
                            note = pretty_midi.Note(
                                velocity=note_velocity,
                                pitch=note_pitch,
                                start=note_time,
                                end=note_end
                            )
                            instrument.notes.append(note)
            
            midi.instruments.append(instrument)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        instrument_names = "-".join([track['family'].lower() for track in st.session_state['instrument_tracks'][:2]])
        filename = f"melody_{instrument_names}_{timestamp}.mid"
        output_path = filename
        
        midi.write(output_path)
        
        progress_placeholder.empty()
        progress_bar.empty()
        
        st.markdown('<div class="success-message">✅ Melody successfully generated!</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## 🎵 Generated Melody")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Melody Details")
            st.markdown(f"""
            - **Instruments:** {", ".join([track['family'] for track in st.session_state['instrument_tracks']])}
            - **Duration:** {duration} seconds
            - **Tempo:** {tempo} BPM
            - **Total Phases:** {sum(len(track['phases']) for track in st.session_state['instrument_tracks'])}
            """)
            
            # Add download button for the MIDI file
            with open(output_path, "rb") as file:
                btn = st.download_button(
                    label="Download MIDI File",
                    data=file,
                    file_name=filename,
                    mime="audio/midi"
                )
        
        with col2:
            # Timeline visualization
            timeline_height = 30 * len(st.session_state['instrument_tracks']) + 20
            timeline_width = 500
            
            svg = f'<svg width="{timeline_width}" height="{timeline_height}" style="background-color: #f1f5f9; border-radius: 6px;">'
            
            for i in range(0, int(duration) + 1, 5):
                x_pos = (i / duration) * (timeline_width - 40) + 20
                svg += f'<line x1="{x_pos}" y1="10" x2="{x_pos}" y2="{timeline_height-10}" stroke="#cbd5e1" stroke-width="1" />'
                svg += f'<text x="{x_pos}" y="10" text-anchor="middle" font-size="10" fill="#64748b">{i}s</text>'
            
            for i, track in enumerate(st.session_state['instrument_tracks']):
                y_pos = 20 + (i * 30)
                svg += f'<text x="10" y="{y_pos+5}" font-size="12" fill="#334155">{track["family"]}</text>'
                
                for phase in sorted(track['phases'], key=lambda x: x['start_time']):
                    start_x = (phase['start_time'] / duration) * (timeline_width - 40) + 20
                    width = ((phase['end_time'] - phase['start_time']) / duration) * (timeline_width - 40)
                    
                    colors = {
                        "Arpeggio": "#3b82f6", "Up-Down": "#ef4444", "Scale": "#10b981",
                        "Simple Loop": "#f59e0b", "Custom": "#8b5cf6"
                    }
                    color = colors.get(phase['melody_type'], "#3b82f6")
                    
                    svg += f'<rect x="{start_x}" y="{y_pos-10}" width="{width}" height="20" rx="3" ry="3" fill="{color}" opacity="0.7" />'
                    
                    if width > 40:
                        svg += f'<text x="{start_x + width/2}" y="{y_pos+4}" text-anchor="middle" font-size="10" fill="white">{phase["melody_type"]}</text>'
            
            svg += '</svg>'
            st.markdown(svg, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error generating melody: {str(e)}")
def load_lottie_url(url: str):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# Main app layout
def main():
  import streamlit as st
from streamlit_lottie import st_lottie
import requests



def load_hero_animation():
    """Load an interactive piano wave visualization for the hero section"""
    html_code = """
   <div id="landing-animation" style="width:100vw; height:100vh; margin:0; padding:0; position:relative; overflow:hidden;">
        <!-- Hero Section with 3D Animation -->
        <div id="hero-section" style="width:100%; height:650px; overflow:hidden; position:relative;">
            <div id="hero-text" style="position:absolute; top:50px; left:50%; transform:translateX(-50%); color:#ffffff; font-family:Arial, sans-serif; text-align:center; z-index:10; text-shadow:0 2px 4px rgba(0,0,0,0.5);">
                <h1 style="font-size:48px; font-weight:800; margin-bottom:12px;">Chorus.AI</h1>
                <p style="font-size:22px; opacity:0.9;">Create AI-powered musical experiences</p>
                 

    <script>
        document.getElementById("get-started-btn").onclick = function() {
            window.clickedTab = 1;  // "⚙️ Settings" tab index
            window.dispatchEvent(new Event("input"));  // Notify Streamlit to rerun
        };
    </script>
            </div>
            <div id="canvas-container"></div>
        </div>
        
        <!-- Features Section -->
        <div id="features-section" style="width:100%; padding:40px 0; background-color:#f8fafc; text-align:center;">
            <h2 style="font-size:36px; color:#0f172a; margin-bottom:40px;">Powerful Features</h2>
            
            <div id="features-grid" style="display:flex; justify-content:center; flex-wrap:wrap; max-width:1200px; margin:0 auto;">
                <!-- Feature 1 -->
                <div class="feature-card" style="width:300px; margin:15px; padding:30px; background-color:white; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.05); transition:transform 0.3s ease;">
                    <div class="feature-icon" style="font-size:36px; margin-bottom:20px;">🎹</div>
                    <h3 style="font-size:22px; color:#0f172a; margin-bottom:12px;">AI Sound Generation</h3>
                    <p style="color:#64748b; line-height:1.6;">Create unique sounds using Google's NSynth neural network technology.</p>
                </div>
                
                <!-- Feature 2 -->
                <div class="feature-card" style="width:300px; margin:15px; padding:30px; background-color:white; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.05); transition:transform 0.3s ease;">
                    <div class="feature-icon" style="font-size:36px; margin-bottom:20px;">🎛️</div>
                    <h3 style="font-size:22px; color:#0f172a; margin-bottom:12px;">Advanced Controls</h3>
                    <p style="color:#64748b; line-height:1.6;">Fine-tune every aspect of your melodies with intuitive controls.</p>
                </div>
                
                <!-- Feature 3 -->
                <div class="feature-card" style="width:300px; margin:15px; padding:30px; background-color:white; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.05); transition:transform 0.3s ease;">
                    <div class="feature-icon" style="font-size:36px; margin-bottom:20px;">🔄</div>
                    <h3 style="font-size:22px; color:#0f172a; margin-bottom:12px;">Multi-Phase Compositions</h3>
                    <p style="color:#64748b; line-height:1.6;">Create evolving music with multiple phases and instrument combinations.</p>
                </div>
            </div>
        </div>
        
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <script>
       // THREE.JS SETUP FOR HERO SECTION
// THREE.JS SETUP WITH OPTIMIZED CODE
(function() {
    // Scene setup
    const heroScene = new THREE.Scene();
    heroScene.background = new THREE.Color(0xe0f2fe);
    
    // Camera setup with responsive FOV
    const heroCamera = new THREE.PerspectiveCamera(65, window.innerWidth / window.innerHeight, 0.1, 1000);
    heroCamera.position.z = 25;
    
    // Renderer setup
    const heroRenderer = new THREE.WebGLRenderer({ antialias: true });
    heroRenderer.setSize(window.innerWidth, window.innerHeight);
    heroRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Limit pixel ratio for performance
    
    // Add renderer to container
    const container = document.getElementById('canvas-container');
    container.appendChild(heroRenderer.domElement);
    
    // Detect mobile for performance optimizations
    const isMobile = window.innerWidth < 768;
    
    // Create vinyl record - MODIFIED: using black and blue colors with increased size
    const createVinyl = () => {
        const group = new THREE.Group();
        // MODIFIED: Increased vinyl scale by 50%
        const scale = isMobile ? 1.2 : 1.5;
        
        // Record disc - MODIFIED: changed color to black
        const record = new THREE.Mesh(
            new THREE.CylinderGeometry(4 * scale, 4 * scale, 0.2, 32),
            new THREE.MeshPhongMaterial({ color: 0x111111 })
        );
        record.rotation.x = Math.PI / 2;
        group.add(record);
        
        // Record label - MODIFIED: deeper blue color
        const label = new THREE.Mesh(
            new THREE.CylinderGeometry(1.3 * scale, 1.3 * scale, 0.21, 32),
            new THREE.MeshPhongMaterial({ color: 0x0284c7 })
        );
        label.rotation.x = Math.PI / 2;
        label.position.z = 0.01;
        group.add(label);
        
        // Simplified grooves - fewer for performance
        const grooveCount = isMobile ? 5 : 8;
        for (let i = 0; i < grooveCount; i++) {
            const groove = new THREE.Mesh(
                new THREE.TorusGeometry(1.6 * scale + i * 0.3 * scale, 0.04, 16, 32),
                new THREE.MeshBasicMaterial({ color: 0x38bdf8 })
            );
            groove.rotation.x = Math.PI / 2;
            group.add(groove);
        }
        
        // Center hole
        const hole = new THREE.Mesh(
            new THREE.CylinderGeometry(0.15 * scale, 0.15 * scale, 0.3, 32),
            new THREE.MeshBasicMaterial({ color: 0x000000 })
        );
        hole.rotation.x = Math.PI / 2;
        group.add(hole);
        
        // MODIFIED: Position vinyl in center
        group.position.y = 0;
        group.position.z = 0;
        
        return group;
    };
    
    // Create all scene elements - MODIFIED: adjusted headphone positioning
    const createElements = () => {
        // Create headphones
        const headphones = new THREE.Group();
        
        // Headband
        const headband = new THREE.Mesh(
            new THREE.TorusGeometry(6.5, 0.6, 16, 32, Math.PI),
            new THREE.MeshPhongMaterial({ color: 0x075985  })
        );
        headband.rotation.x = Math.PI;
        headband.position.y = 1;
        headphones.add(headband);
        
        // Ear cups and pads
        [-6.5, 6.5].forEach(x => {
            // Cup
            const cup = new THREE.Mesh(
                new THREE.CylinderGeometry(3.2, 3.2, 1.8, 32),
                new THREE.MeshPhongMaterial({ color: 0x0284c7 })
            );
            cup.rotation.z = Math.PI / 2;
            cup.position.x = x;
            headphones.add(cup);
            
            // Pad
            const pad = new THREE.Mesh(
                new THREE.TorusGeometry(2.2, 1, 16, 32),
                new THREE.MeshPhongMaterial({ color: 0xbae6fd })
            );
            pad.position.x = x;
            pad.rotation.y = Math.PI / 2;
            headphones.add(pad);
        });
        
        // Cord
        const curve = new THREE.CubicBezierCurve3(
            new THREE.Vector3(-6.5, 0, 0),
            new THREE.Vector3(-4, -4, 2),
            new THREE.Vector3(-2.5, -6, 0),
            new THREE.Vector3(0, -8, 0)
        );
        
        const points = curve.getPoints(20);
        const cord = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(points),
            new THREE.LineBasicMaterial({ color: 0x0369a1, linewidth: 5 })
        );
        headphones.add(cord);
        
        // MODIFIED: Position headphones over vinyl
        headphones.position.y = 5;
        headphones.position.z = 3;
        const scale = isMobile ? 1 : 1.3;
        headphones.scale.set(scale, scale, scale);
        
        return { headphones };
    };
    
    // Create equalizer with fewer bars for mobile
    const createEqualizer = () => {
        const group = new THREE.Group();
        const barCount = isMobile ? 8 : 12;
        const barWidth = isMobile ? 0.8 : 0.6;
        const barSpacing = isMobile ? 1.1 : 0.9;
        const materials = [
            new THREE.MeshPhongMaterial({ color: 0x0ea5e9 }),
            new THREE.MeshPhongMaterial({ color: 0x0284c7 }),
            new THREE.MeshPhongMaterial({ color: 0x0369a1 })
        ];
        
        for (let i = 0; i < barCount; i++) {
            const height = Math.random() * 3 + 1;
            const bar = new THREE.Mesh(
                new THREE.BoxGeometry(barWidth, height, barWidth),
                materials[i % materials.length]
            );
            bar.position.x = i * barSpacing - (barCount * barSpacing / 2) + (barSpacing / 2);
            bar.position.y = height / 2 - 5;
            bar.userData = {
                originalHeight: height,
                targetHeight: height,
                speed: 0.05 + Math.random() * 0.1
            };
            group.add(bar);
        }
        
        // MODIFIED: Move equalizer down to accommodate centered vinyl
        group.position.y = -6;
        
        return group;
    };
    
    // Create background lines with fewer for mobile
    const createBackgroundLines = () => {
        const group = new THREE.Group();
        const lineCount = isMobile ? 12 : 20;
        const blueColors = [0xbae6fd, 0x7dd3fc, 0x38bdf8, 0x0ea5e9];
        
        for (let i = 0; i < lineCount; i++) {
            const segments = Math.floor(Math.random() * 3) + 3; // 3-5 segments for better performance
            const widthFactor = isMobile ? 15 : 20;
            const startX = Math.random() * (widthFactor * 2) - widthFactor;
            const startY = Math.random() * 30 - 15;
            const startZ = Math.random() * 15 - 20;
            
            const linePoints = [];
            for (let j = 0; j <= segments; j++) {
                const x = startX + j * (Math.random() * 0.8 + 0.7);
                const y = startY + Math.sin(j * 0.5) * (Math.random() * 0.8 + 0.7);
                linePoints.push(new THREE.Vector3(x, y, startZ));
            }
            
            const lineGeometry = new THREE.BufferGeometry().setFromPoints(linePoints);
            const color = blueColors[Math.floor(Math.random() * blueColors.length)];
            const line = new THREE.Line(
                lineGeometry,
                new THREE.LineBasicMaterial({ 
                    color: color,
                    transparent: true,
                    opacity: 0.3 + Math.random() * 0.3
                })
            );
            
            line.userData = {
                originalPoints: linePoints.map(p => ({ x: p.x, y: p.y, z: p.z })),
                speed: 0.005 + Math.random() * 0.01,
                offset: Math.random() * Math.PI * 2
            };
            
            group.add(line);
        }
        
        return group;
    };
    
    // Create containers for dynamic elements
    const musicNotes = new THREE.Group();
    
    // Create scene elements
    const vinyl = createVinyl();
    const { headphones } = createElements();
    const equalizer = createEqualizer();
    const backgroundLines = createBackgroundLines();
    
    // Add created elements to scene
    heroScene.add(vinyl, headphones, equalizer, musicNotes, backgroundLines);
    
    // Add lights - simplified lighting setup
    heroScene.add(new THREE.AmbientLight(0xffffff, 0.8));
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2);
    directionalLight.position.set(5, 5, 5);
    heroScene.add(directionalLight);
    
    // Animation variables
    let time = 0;
    let noteTimer = 0;
    
    // Create note function - simplified from original
    const createNote = () => {
        const group = new THREE.Group();
        const material = new THREE.MeshBasicMaterial({ 
            color: 0x0ea5e9,
            transparent: true,
            opacity: 1 
        });
        
        // Just create a simple note head for performance
        const noteHead = new THREE.Mesh(
            new THREE.SphereGeometry(0.4, 16, 12),
            material
        );
        noteHead.scale.y = 0.7;
        group.add(noteHead);
        
        // Simple stem
        const stem = new THREE.Mesh(
            new THREE.BoxGeometry(0.09, 1.3, 0.09),
            material
        );
        stem.position.y = 0.65;
        stem.position.x = 0.25;
        group.add(stem);
        
        return group;
    };
    
    // Animation function - simplified for performance
    function animate() {
        requestAnimationFrame(animate);
        time += 0.01;
        
        // MODIFIED: Gentle floating animation for headphones above vinyl
        headphones.position.y = 5 + Math.sin(time) * 0.2;
        
        // Simplified equalizer animation
        equalizer.children.forEach(bar => {
            // Set new target less frequently for performance
            if (Math.random() < 0.03) {
                bar.userData.targetHeight = Math.random() * 3 + 1;
            }
            
            // Move towards target height
            bar.scale.y += (bar.userData.targetHeight - bar.scale.y) * 0.1;
        });
        
        // Rotate vinyl
        vinyl.rotation.z += 0.01;
        
        // Spawn notes less frequently on mobile
        noteTimer -= 0.016;
        if (noteTimer <= 0) {
            noteTimer = isMobile ? 0.5 : 0.2;
            
            if (musicNotes.children.length < (isMobile ? 10 : 20)) {
                const note = createNote();
                const posRange = isMobile ? 8 : 12;
                note.position.x = Math.random() * (posRange * 2) - posRange;
                note.position.y = -5;
                note.position.z = Math.random() * 4 - 2;
                
                note.userData = {
                    age: 0,
                    maxAge: 4,
                    speed: 0.05
                };
                
                musicNotes.add(note);
            }
        }
        
        // Animate existing notes
        musicNotes.children.forEach((note, index) => {
            note.userData.age += 0.016;
            note.position.y += note.userData.speed;
            
            // Fade out based on age
            const opacity = 1 - (note.userData.age / note.userData.maxAge);
            note.children.forEach(child => {
                if (child.material) {
                    child.material.opacity = opacity;
                }
            });
            
            // Remove old notes
            if (note.userData.age >= note.userData.maxAge) {
                musicNotes.remove(note);
            }
        });
        
        // Simplified background line animation - only animate every other frame for performance
        if (time % 0.02 < 0.01) {
            backgroundLines.children.forEach(line => {
                const positions = line.geometry.attributes.position;
                const originalPoints = line.userData.originalPoints;
                
                for (let i = 0; i < positions.count; i++) {
                    const y = originalPoints[i].y + Math.sin(time + i * 0.3) * 0.4;
                    positions.array[i * 3 + 1] = y;
                }
                
                positions.needsUpdate = true;
            });
        }
        
        // Render scene
        heroRenderer.render(heroScene, heroCamera);
    }
    
    // Handle window resize
    function onResize() {
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        heroCamera.aspect = width / height;
        heroCamera.updateProjectionMatrix();
        heroRenderer.setSize(width, height);
    }
    
    window.addEventListener('resize', onResize);
    
    // Start animation
    animate();
})();

// D3 Workflow Visualization - Simplified
(function() {
    const workflowSteps = [
        { icon: "🎹", label: "Select Instruments" },
        { icon: "🎛️", label: "Adjust Parameters" },
        { icon: "✨", label: "Generate Melody" },
        { icon: "🔊", label: "Fine-tune & Export" }
    ];
    
    const svg = d3.select("#workflow-visualization")
        .append("svg")
        .attr("width", "100%")
        .attr("height", "100%")
        .attr("viewBox", "0 0 800 150");
        
    // Create responsive step positions
    const stepWidth = Math.min(170, window.innerWidth / 5);
    const startX = (800 - (workflowSteps.length - 1) * stepWidth) / 2;
    
    // Create groups for steps
    const steps = svg.selectAll(".step")
        .data(workflowSteps)
        .enter()
        .append("g")
        .attr("class", "step")
        .attr("transform", (d, i) => `translate(${startX + i * stepWidth}, 60)`);
        
    // Add circles
    steps.append("circle")
        .attr("r", 30)
        .attr("fill", "#ec4899")
        .attr("stroke", "#fff")
        .attr("stroke-width", 2);
        
    // Add icons
    steps.append("text")
        .attr("text-anchor", "middle")
        .attr("y", 5)
        .attr("fill", "white")
        .attr("font-size", "16px")
        .attr("font-weight", "bold")
        .text(d => d.icon);
        
    // Add labels
    steps.append("text")
        .attr("text-anchor", "middle")
        .attr("y", 60)
        .attr("fill", "white")
        .attr("font-size", "14px")
        .text(d => d.label);
        
    // Add connecting lines
    svg.selectAll(".connection")
        .data(workflowSteps.slice(0, -1))
        .enter()
        .append("line")
        .attr("x1", (d, i) => startX + i * stepWidth + 40)
        .attr("y1", 60)
        .attr("x2", (d, i) => startX + (i+1) * stepWidth - 40)
        .attr("y2", 60)
        .attr("stroke", "#ec4899")
        .attr("stroke-width", 3)
        .attr("stroke-dasharray", "5,5");
        
    // Simple pulse animation for steps
    steps.select("circle")
        .transition()
        .duration(2000)
        .attr("r", 35)
        .transition()
        .duration(2000)
        .attr("r", 30)
        .on("end", function repeat() {
            d3.select(this)
                .transition()
                .duration(2000)
                .attr("r", 35)
                .transition()
                .duration(2000)
                .attr("r", 30)
                .on("end", repeat);
        });
})();

// Feature card animations
(function() {
    const featureCards = document.querySelectorAll('.feature-card');
    
    // Add hover effects
    featureCards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-10px)';
            card.style.boxShadow = '0 10px 25px rgba(0,0,0,0.1)';
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0)';
            card.style.boxShadow = '0 4px 12px rgba(0,0,0,0.05)';
        });
        
        // Initially hide cards
        card.style.transform = 'translateY(20px)';
        card.style.opacity = '0';
        card.style.transition = 'transform 0.5s ease, opacity 0.5s ease, box-shadow 0.3s ease';
    });
    
    // Initial animation when page loads
    setTimeout(() => {
        featureCards.forEach((card, index) => {
            setTimeout(() => {
                card.style.transform = 'translateY(0)';
                card.style.opacity = '1';
            }, index * 100);
        });
    }, 500);
    
    // Get started button effects
    const getStartedBtn = document.getElementById('get-started-btn');
    getStartedBtn.addEventListener('mousedown', () => getStartedBtn.style.transform = 'scale(0.95)');
    getStartedBtn.addEventListener('mouseup', () => getStartedBtn.style.transform = 'scale(1)');
    getStartedBtn.addEventListener('mouseleave', () => getStartedBtn.style.transform = 'scale(1)');
    
    // Scroll animations
    const handleScroll = () => {
        featureCards.forEach((card, index) => {
            const rect = card.getBoundingClientRect();
            const isVisible = rect.top < window.innerHeight && rect.bottom >= 0;
            
            if (isVisible && card.style.opacity < '1') {
                setTimeout(() => {
                    card.style.transform = 'translateY(0)';
                    card.style.opacity = '1';
                }, index * 100);
            }
        });
    };
    
    
    window.addEventListener('scroll', handleScroll);
    // Initial check
    handleScroll();
})();
    </script>
    """
    return html_code

def main():
    
    init_session_state()
    
    # Load the landing page animation
    components.html(load_hero_animation(), height=1100)  # Reduced height to minimize gap
    
    # Apply custom CSS with the updated styles
    st.markdown("""
    <style>
    .main-content {
        max-width: 1200px;
        margin: auto;
        padding: 2rem;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin-top: 2rem;
    }
    
    .feature-item {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    
    .feature-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: #ec4899;
    }
    
    .feature-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        color: #64748b;
        line-height: 1.6;
    }
    
    .cta-section {
        background: linear-gradient(145deg, #0f172a, #1e293b);
        padding: 3rem 2rem;
        border-radius: 16px;
        color: #ffffff;
        text-align: center;
        margin: 3rem auto;
        max-width: 1000px;
    }
    
    .cta-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .cta-subtitle {
        font-size: 1.1rem;
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    
    .cta-button {
        display: inline-block;
        background: #ec4899;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .cta-button:hover {
        background: #be185d;
        transform: translateY(-2px);
    }
    
    /* Fix for the gap between hero section and tabs */
    .main .block-container {
        max-width: 100%;
        padding-left: 1;
        padding-right: 1;
        padding-top: 0 !important;
        margin-top: -5px !important; /* Close the gap */
    }
    
    
    /* Ensure the landing animation container fits properly */
    #landing-animation {
        margin-bottom: 0 !important; 
        height: 95vh !important; /* Slightly shorter to avoid scrollbar issues */
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Tabs for the app - with a custom ID for JavaScript targeting
    tabs = st.tabs(["✏️ Compose", "⚙️ Settings"])
    
    with tabs[0]:
        compose_tab()
    
    with tabs[1]:
        settings_tab()
    
    





if __name__ == "__main__":
    main()
