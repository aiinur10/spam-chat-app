# train_model.py
import os                               # modul untuk mengelola path direktori
import pandas as pd                     # library untuk membaca & manipulasi dataset
from sklearn.feature_extraction.text import TfidfVectorizer  # mengubah teks jadi angka menggunakan TF-IDF
from sklearn.naive_bayes import MultinomialNB               # algoritma machine learning Naive Bayes
from sklearn.model_selection import train_test_split        # membagi dataset menjadi train & test
from sklearn.metrics import classification_report, accuracy_score  # evaluasi model
import joblib                          # untuk menyimpan model ke file .pkl

# menetapkan direktori file dataset dan lokasi penyimpanan model
BASE = os.path.dirname(__file__)                      # path folder file ini berada
DATA = os.path.join(BASE, "data.csv")                 # file dataset
MODEL_DIR = os.path.join(BASE, "model")               # folder untuk menyimpan model
os.makedirs(MODEL_DIR, exist_ok=True)                 # membuat folder model jika belum ada
MODEL_PATH = os.path.join(MODEL_DIR, "model_sms.pkl") # file tempat menyimpan model
VEC_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")  # file tempat menyimpan vectorizer

def load_data(path):
    df = pd.read_csv(path)                            # membaca file CSV dataset
    if 'label' not in df.columns or 'text' not in df.columns:
        # dataset harus berisi kolom label (spam/ham) dan text (isi pesan)
        raise ValueError("Dataset harus punya kolom 'label' dan 'text'")
    return df['text'], df['label']                    # mengembalikan isi pesan & label

def train():
    print("Loading dataset:", DATA)
    X, y = load_data(DATA)                            # memanggil fungsi load dataset

    vect = TfidfVectorizer(min_df=1, ngram_range=(1,2))  # TF-IDF untuk mengubah teks jadi angka
    Xv = vect.fit_transform(X)                           # melakukan training TF-IDF pada teks

    # Membagi dataset menjadi 80% data latih dan 20% data uji
    Xtr, Xte, ytr, yte = train_test_split(Xv, y, test_size=0.2, random_state=42, stratify=y)

    model = MultinomialNB()                         # memilih algoritma ML Multinomial Naive Bayes
    model.fit(Xtr, ytr)                             # melatih model menggunakan data latih

    preds = model.predict(Xte)                      # prediksi pada data uji
    print("Accuracy:", accuracy_score(yte, preds))  # akurasi model
    print(classification_report(yte, preds, zero_division=0))  # evaluasi detail

    joblib.dump(model, MODEL_PATH)                  # menyimpan model ke file .pkl
    joblib.dump(vect, VEC_PATH)                     # menyimpan vectorizer ke file .pkl
    print("Saved model ->", MODEL_PATH)
    print("Saved vectorizer ->", VEC_PATH)

# menjalankan fungsi train() jika file ini dieksekusi langsung
if __name__ == "__main__":
    train()
