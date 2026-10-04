"""app.py - Flask backend for the Fake News Detection System."""
import json
import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

from preprocess import preprocess_steps

BASE = os.path.dirname(os.path.abspath(__file__))
MIN_WORDS, MAX_CHARS = 20, 50_000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

model = vectorizer = None
metrics = {}


def load_artifacts():
    global model, vectorizer, metrics
    try:
        model = joblib.load(os.path.join(BASE, "model.pkl"))
        vectorizer = joblib.load(os.path.join(BASE, "vectorizer.pkl"))
        with open(os.path.join(BASE, "assets", "metrics.json")) as f:
            metrics = json.load(f)
    except FileNotFoundError:
        pass


load_artifacts()


def explain(steps, tfidf_row, top_n=10):
    """Rank input terms by their contribution (coefficient x tf-idf weight)."""
    vocab = vectorizer.vocabulary_
    coefs = model.coef_[0]
    # map each stem back to the first original word it came from
    stem_to_word = {}
    for word, st in zip(steps["no_stopwords"], steps["stemmed"]):
        stem_to_word.setdefault(st, word)
    out = []
    for st, word in stem_to_word.items():
        idx = vocab.get(st)
        if idx is None:
            continue
        w = float(tfidf_row[0, idx])
        if w == 0:
            continue
        out.append({"word": word, "stem": st,
                    "weight": round(float(coefs[idx]) * w, 4),
                    "tfidf": round(w, 4)})
    out.sort(key=lambda d: abs(d["weight"]), reverse=True)
    return out[:top_n]


@app.route("/")
def home():
    return render_template("index.html", metrics=metrics, ready=model is not None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify(error="Model files not found. Run 'python model.py' first."), 503
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        return jsonify(error="Send JSON like {\"text\": \"...\"}."), 400
    text = data["text"].strip()
    if not text:
        return jsonify(error="Please paste a news article first."), 400
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Article is too long (max {MAX_CHARS:,} characters)."), 400
    if len(text.split()) < MIN_WORDS:
        return jsonify(error=f"Article is too short. Enter at least {MIN_WORDS} words."), 400

    steps = preprocess_steps(text)
    if not steps["stemmed"]:
        return jsonify(error="No meaningful words found in the text."), 400

    row = vectorizer.transform([" ".join(steps["stemmed"])])
    proba = model.predict_proba(row)[0]          # [P(fake), P(real)]
    is_real = bool(proba[1] >= 0.5)
    keywords = explain(steps, row)
    for k in keywords:
        k["pushes"] = "Real" if k["weight"] > 0 else "Fake"

    return jsonify(
        label="Real News" if is_real else "Fake News",
        confidence=round(float(max(proba)) * 100, 2),
        probabilities={"fake": round(float(proba[0]) * 100, 2),
                       "real": round(float(proba[1]) * 100, 2)},
        keywords=keywords,
        preprocessing={
            "original_words": len(text.split()),
            "lowercased": steps["lowercased"][:600],
            "no_punctuation": steps["no_punctuation"][:600],
            "tokens": steps["tokens"][:60],
            "no_stopwords": steps["no_stopwords"][:60],
            "stemmed": steps["stemmed"][:60],
            "counts": {"tokens": len(steps["tokens"]),
                       "no_stopwords": len(steps["no_stopwords"]),
                       "stemmed": len(steps["stemmed"])},
        },
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Request too large."), 413


@app.errorhandler(404)
def not_found(_):
    return jsonify(error="Not found."), 404


@app.errorhandler(500)
def server_error(_):
    return jsonify(error="Something went wrong on the server."), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
