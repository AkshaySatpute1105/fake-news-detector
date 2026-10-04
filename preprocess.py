"""
preprocess.py - Text preprocessing pipeline (Information Retrieval step 1).

Pipeline: lowercase -> punctuation removal -> tokenization
          -> stop-word removal -> Porter stemming
No NLTK data downloads are required (regex tokenizer + scikit-learn stop-word list).
"""
import re
import string
from functools import lru_cache

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_stemmer = PorterStemmer()
STOP_WORDS = set(ENGLISH_STOP_WORDS)
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation + "“”‘’—–…"})


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    return _stemmer.stem(word)


def preprocess_steps(text: str) -> dict:
    """Return every intermediate preprocessing stage (used by the UI)."""
    lowered = text.lower()
    no_punct = re.sub(r"\s+", " ", lowered.translate(_PUNCT_TABLE)).strip()
    tokens = re.findall(r"[a-z0-9]+", no_punct)
    no_stop = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    stemmed = [stem(t) for t in no_stop]
    return {
        "lowercased": lowered,
        "no_punctuation": no_punct,
        "tokens": tokens,
        "no_stopwords": no_stop,
        "stemmed": stemmed,
    }


def clean_text(text: str) -> str:
    """Return the fully preprocessed text as a single string (fed to TF-IDF)."""
    return " ".join(preprocess_steps(text)["stemmed"])
