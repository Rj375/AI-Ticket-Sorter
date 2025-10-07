# app/app.py
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path


ARTIFACTS = Path('artifacts')
df = pd.read_csv("data/tickets_clean.csv")

@st.cache_resource
def load_artifacts():
    clf = joblib.load(ARTIFACTS / 'model.pkl')
    vec = joblib.load(ARTIFACTS / 'vectorizer.pkl')
    return clf, vec


clf, vec = load_artifacts()


st.set_page_config(page_title='AI Ticket Sorter', layout='centered')
st.title('AI Ticket Sorter')

text = st.text_area('Paste ticket text in the message box below')

st.button('Predict')

# simple cleaning (must match train preprocess)
import re
def clean_text(s):
        s = str(s).lower()
        s = re.sub(r"[^a-z0-9\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
text_clean = clean_text(text)
X = vec.transform([text_clean])
pred = clf.predict(X)[0]

#conditional statement For prediction
if not text.strip():
    st.error('Please provide text')
else:
     st.success(f'Prediction: **{pred}**') 


