# Language Detection System

A machine-learning system that identifies the language of a text snippet using
**character n-gram TF-IDF features** and classical classifiers (Multinomial Naive Bayes,
Logistic Regression, Linear SVM). Supports 16 languages:
English, French, German, Spanish, Italian, Portuguese, Dutch, Swedish, Polish, Turkish,
Russian, Greek, Hindi, Bengali, Tamil, Kannada.

**Dataset:** Universal Declaration of Human Rights (NLTK `udhr` corpus), split into
1,638 short samples (25-280 characters).

## Results (20% stratified hold-out, 328 samples)
| Model | Accuracy | Macro F1 |
|---|---|---|
| Multinomial Naive Bayes (best) | 99.39% | 0.996 |
| Linear SVM | 99.09% | 0.994 |
| Logistic Regression | 98.78% | 0.991 |

Plots are in `results/`, full metrics in `results/metrics.json`.

## Setup & run
```bash
pip install -r requirements.txt

# (optional) rebuild dataset - needs the raw UDHR files in data/udhr/
# curl -L -o udhr.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/udhr.zip
# unzip udhr.zip -d data/
python src/prepare_data.py

python src/detector.py train
python src/detector.py predict "Where is the nearest railway station?"
```

## Project structure
```
src/prepare_data.py   build dataset.csv from raw UDHR files
src/detector.py       training, evaluation, plots, prediction CLI
data/dataset.csv      labelled samples
results/              metrics.json + figures
model.joblib          trained model
```

## Limitations
- Only the 16 trained languages are recognised; unseen languages (e.g. Malayalam) get a wrong, low-confidence label.
- Test samples come from the same 16 documents as training samples, so accuracy is optimistic for other text domains.
- Very short inputs (a word or two) and closely related languages (Spanish/Portuguese) are the main error sources.
