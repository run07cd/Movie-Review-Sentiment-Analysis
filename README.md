# Movie Review Sentiment Analysis

A Python machine-learning project that predicts whether a movie review is positive or negative.

## Run it

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

On the first training run, the script downloads the movie review dataset through NLTK. After training, open the local Streamlit URL printed in the terminal.

## How it works

- Uses the labeled movie reviews corpus included with NLTK.
- Cleans basic HTML and whitespace in the review text.
- Converts text into TF-IDF features, including single words and two-word phrases.
- Trains Logistic Regression to classify each review.
- Evaluates on a held-out, stratified test set using accuracy, precision, recall, F1, and a confusion matrix.
- Saves the trained pipeline and test metrics in the local `artifacts` folder.
- Provides a small Streamlit app to test a review and view evaluation results.

## Files

- `sentiment_model.py` — text cleaning and model pipeline.
- `train_model.py` — dataset loading, training, evaluation, and saving.
- `app.py` — interactive app.
- `requirements.txt` — Python dependencies.

## Notes

The dataset contains positive and negative movie reviews; it does not include a neutral class. The model is a traditional TF-IDF + Logistic Regression baseline, not a large language model. Results on movie reviews may not transfer to sarcasm, short social-media posts, or other domains. Model probabilities are estimates and should not be treated as guaranteed confidence.
