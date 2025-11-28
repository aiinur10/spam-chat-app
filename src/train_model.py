# train_model.py
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data.csv")  # atur nama file dataset atau path
MODEL_DIR = os.path.join(BASE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "model_sms.pkl")
VEC_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

def load_data(path):
    df = pd.read_csv(path)
    if 'label' not in df.columns or 'text' not in df.columns:
        raise ValueError("Dataset harus punya kolom 'label' dan 'text'")
    return df['text'], df['label']

def train():
    print("Loading dataset:", DATA)
    X, y = load_data(DATA)

    vect = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    Xv = vect.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(Xv, y, test_size=0.2, random_state=42, stratify=y)
    model = MultinomialNB()
    model.fit(Xtr, ytr)

    preds = model.predict(Xte)
    print("Accuracy:", accuracy_score(yte, preds))
    print(classification_report(yte, preds, zero_division=0))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vect, VEC_PATH)
    print("Saved model ->", MODEL_PATH)
    print("Saved vectorizer ->", VEC_PATH)

if __name__ == "__main__":
    train()
