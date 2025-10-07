# app/app.py
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path


ARTIFACTS = Path('artifacts')


@st.cache_resource
def load_artifacts():
    clf = joblib.load(ARTIFACTS / 'model.pkl')
    vec = joblib.load(ARTIFACTS / 'vectorizer.pkl')
    return clf, vec


clf, vec = load_artifacts()


st.set_page_config(page_title='AI Ticket Sorter', layout='centered')
st.title('AI Ticket Sorter')
# st.write('Classify support tickets into categories')


# mode = st.radio('Mode', ['Single ticket', 'Batch (CSV)'])
# mode = st.radio('Mode', ['Batch (CSV)'])

text = st.text_area('Paste ticket text here')
# if mode == 'Single ticket':
if st.button('Predict'):
    if not text.strip():
        st.error('Please provide text')
else:
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
    proba = clf.predict_proba(X)[0]
    classes = clf.classes_
    st.success(f'Prediction: **{pred, classes}**')
    # st.write('Probabilities:')
    # classes = clf.classes_
    # dfp = pd.DataFrame({'category': classes, 'prob': proba}).sort_values('prob', ascending=False)
    # st.dataframe(dfp)



# uploaded = st.file_uploader('Upload CSV with a `text` column', type=['csv'])
# if uploaded is not None:
    df = pd.read_csv("data/tickets_clean.csv")
#     st.success("✅ File uploaded successfully!")
#     st.write("Preview of data:", df.head())

    if 'text' not in df.columns:
        st.error('CSV must contain a `text` column')
    else:
        import re
        df['text_clean'] = df['text'].astype(str).apply(lambda s: re.sub(r"[^a-z0-9\s]", "", s.lower()))
        X = vec.transform(df['text_clean'])
        df['pred'] = clf.predict(X)
        # st.download_button('Download predictions', df.to_csv(index=False), file_name='predictions.csv')
        # st.dataframe(df.head(50))
# else:
#     st.warning("⚠️ Please upload a CSV file to continue.")


