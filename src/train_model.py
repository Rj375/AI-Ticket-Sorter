import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# load cleaned data
df = pd.read_csv('/Users/rameshgiri/Desktop/ai-ticket-sorter/data/tickets_clean.csv')
X = df['text_clean']
y = df['category']


# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


# vectorize
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# model
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train_tfidf, y_train)


# predict
y_pred = clf.predict(X_test_tfidf)


# eval
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# save artifacts
joblib.dump(clf, '/Users/rameshgiri/Desktop/ai-ticket-sorter/artifacts/model.pkl')
joblib.dump(vectorizer, '/Users/rameshgiri/Desktop/ai-ticket-sorter/artifacts/vectorizer.pkl')