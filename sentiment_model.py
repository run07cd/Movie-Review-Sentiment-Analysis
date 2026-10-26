import html
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def normalize_review(text):
    """Normalize basic HTML and whitespace while preserving useful words."""
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_reviews(reviews):
    return [normalize_review(review) for review in reviews]


def build_model():
    """Build a reproducible TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        [
            ("clean_text", FunctionTransformer(normalize_reviews, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    C=2.0,
                    random_state=42,
                ),
            ),
        ]
    )
