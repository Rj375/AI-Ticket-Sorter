# AI Ticket Sorter


## Overview
A simple ticket classification app using TF-IDF + Logistic Regression and Streamlit for UI.


## Quick Start
1. Clone repo
2. Create virtualenv and install requirements: `pip install -r requirements.txt`
3. Place dataset in `data/` and run training: `python src/train_model.py`
4. Start app: `streamlit run app/app.py`


## Files
- `data/`: raw and cleaned datasets
- `artifacts/`: saved model & vectorizer
- `app/`: Streamlit app
- `notebooks/`: EDA and modeling notebooks


## Notes
- Ensure preprocessing in app matches training.
- For larger data, consider using a smaller `max_features` or feature selection.