**Features**
- Predicts mushroom edibility with a confidence score
- Accepts both numeric inputs (cap diameter, stem dimensions) and categorical selections (cap color, gill attachment, etc.)
- Encodes features to match model training format using one-hot encoding
- Displays real-time prediction and confidence percentage
- Clean, user-friendly UI powered by Streamlit

**Tech Stack**
- Python
- Streamlit
- Pandas, NumPy
- scikit-learn
- Pickle & Joblib (for loading trained model and feature set)

**Files**
- rf_model.pkl — trained Random Forest classifier
- rf_features.pkl — list of all features expected by the model
- app.py — Streamlit app code

**Run the App Locally**

pip install -r requirements.txt

streamlit run app.py
