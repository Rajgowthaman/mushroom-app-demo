import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os

# Load model and features
model_path = os.path.join(os.path.dirname(__file__), 'rf_model.pkl')
model = pickle.load(open(model_path, 'rb'))
model = pickle.load(open("rf_model.pkl", "rb"))
features = joblib.load(open("rf_features.pkl", "rb"))

st.title("Mushroom Edibility Predictor")
st.write("Select mushroom features below to predict if it's edible or poisonous.")

# === NUMERIC FEATURES ===
cap_diameter = st.slider("Cap Diameter (cm)", 1.0, 20.0, 10.0)
stem_height = st.slider("Stem Height (cm)", 1.0, 30.0, 15.0)
stem_width = st.slider("Stem Width (mm)", 1.0, 30.0, 10.0)

# === HUMAN-READABLE DROPDOWNS WITH CODE MAPPING ===
cap_color_options = {
    'White': 'w',
    'Yellow': 'y',
    'Brown': 'n',
    'Red': 'e',
    'Purple': 'u',
    'Black': 'k'
}

does_bruise_options = {
    'No': 'f',
    'Yes': 't'
}

gill_attachment_options = {
    'Adnate': 'a',
    'Decurrent': 'd',
    'Free': 'e',
    'None': 'f',
    'Pores': 'p',
    'Sinuate': 's',
    'Adnexed': 'x'
}

gill_color_options = {
    'Brown': 'n',
    'Buff': 'b',
    'Grey': 'g',
    'Pink': 'p',
    'White': 'w',
    'Yellow': 'y',
    'Black': 'k',
    'Red': 'e'
}

# User inputs
cap_color_label = st.selectbox("Cap Color", list(cap_color_options.keys()))
does_bruise_label = st.selectbox("Does it Bruise or Bleed?", list(does_bruise_options.keys()))
gill_attachment_label = st.selectbox("Gill Attachment", list(gill_attachment_options.keys()))
gill_color_label = st.selectbox("Gill Color", list(gill_color_options.keys()))

# Convert labels to codes
cap_color = cap_color_options[cap_color_label]
does_bruise = does_bruise_options[does_bruise_label]
gill_attachment = gill_attachment_options[gill_attachment_label]
gill_color = gill_color_options[gill_color_label]

# === INPUT DICTIONARY ===
input_dict = {
    'cap-diameter': cap_diameter,
    'stem-height': stem_height,
    'stem-width': stem_width,
    f'cap-color_{cap_color}': 1,
    f'does-bruise-or-bleed_{does_bruise}': 1,
    f'gill-attachment_{gill_attachment}': 1,
    f'gill-color_{gill_color}': 1
}

# === PREDICTION PIPELINE ===
df_input = pd.DataFrame([input_dict])

# Fill in all other expected features with 0
for col in features:
    if col not in df_input.columns:
        df_input[col] = 0

# Ensure column order matches training
df_input = df_input[features]

# Run prediction
prediction = model.predict(df_input)[0]
confidence = model.predict_proba(df_input)[0][int(prediction)]

# Display result
st.subheader("Prediction:")
st.write("Poisonous 🍄" if prediction == 1 else "Edible ✅")
st.write(f"Confidence Score: {confidence:.2%}")