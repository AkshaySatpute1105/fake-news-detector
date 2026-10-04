"""
model.py - Train the Fake News classifier.

    python model.py

Reads  dataset/Fake.csv and dataset/True.csv
Writes model.pkl, vectorizer.pkl, assets/metrics.json, assets/confusion_matrix.png
"""
import json
import os
import re
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

from preprocess import clean_text

BASE = os.path.dirname(os.path.abspath(__file__))
FAKE_CSV = os.path.join(BASE, "dataset", "Fake.csv")
TRUE_CSV = os.path.join(BASE, "dataset", "True.csv")
MODEL_PATH = os.path.join(BASE, "model.pkl")
VEC_PATH = os.path.join(BASE, "vectorizer.pkl")
METRICS_PATH = os.path.join(BASE, "assets", "metrics.json")
CM_PATH = os.path.join(BASE, "assets", "confusion_matrix.png")

# The True.csv articles begin with "CITY (Reuters) -". That tag leaks the label,
# so it is removed to force the model to learn from the writing itself.
REUTERS_RE = re.compile(r"^.{0,120}?\(Reuters\)\s*-?\s*", re.IGNORECASE)


def load_data() -> pd.DataFrame:
    for p in (FAKE_CSV, TRUE_CSV):
        if not os.path.exists(p):
            sys.exit(f"Missing {p}. Download the Kaggle 'Fake and Real News Dataset' "
                     "and place Fake.csv and True.csv in the dataset/ folder.")
    fake = pd.read_csv(FAKE_CSV)
    true = pd.read_csv(TRUE_CSV)
    fake["label"] = 0   # 0 = Fake
    true["label"] = 1   # 1 = Real
    df = pd.concat([fake, true], ignore_index=True)
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("").map(lambda t: REUTERS_RE.sub("", t))
    df["content"] = (df["title"] + " " + df["text"]).str.strip()
    df = df[df["content"].str.len() > 0].drop_duplicates(subset="content")
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    print("Loading data ...")
    df = load_data()
    print(f"  {len(df)} articles  (Fake={int((df.label == 0).sum())}, Real={int((df.label == 1).sum())})")

    print("Preprocessing (lowercase, punctuation, tokenize, stop-words, stemming) ...")
    df["clean"] = df["content"].map(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    print("Building TF-IDF vector space ...")
    vectorizer = TfidfVectorizer(max_features=50000, min_df=2, max_df=0.8,
                                 sublinear_tf=True, ngram_range=(1, 1))
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    print("Training Logistic Regression ...")
    model = LogisticRegression(max_iter=1000, C=3.0, solver="liblinear")
    model.fit(Xtr, y_train)

    pred = model.predict(Xte)
    cm = confusion_matrix(y_test, pred)
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1_score": round(f1_score(y_test, pred), 4),
        "confusion_matrix": cm.tolist(),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
    }
    print(json.dumps(metrics, indent=2))

    os.makedirs(os.path.join(BASE, "assets"), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    try:  # confusion-matrix image (optional)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["Fake", "Real"]); ax.set_yticks([0, 1], ["Fake", "Real"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        fig.tight_layout(); fig.savefig(CM_PATH, dpi=150); plt.close(fig)
    except Exception as e:  # pragma: no cover
        print("Could not draw confusion matrix:", e)

    print(f"Saved {MODEL_PATH}\nSaved {VEC_PATH}")


if __name__ == "__main__":
    main()
