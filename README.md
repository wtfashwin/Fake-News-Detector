# Fake News Detector

Fake News Detector is a learning notebook for training a text classifier on news articles labeled as fake or real.

## What it does

`Fake_news_detection.ipynb` cleans article text, removes English stop words, lemmatizes words, creates TF-IDF features, and trains a `PassiveAggressiveClassifier`. It includes an 80/20 split, accuracy and classification-report calculations, a confusion-matrix plot, and interactive text prediction. Training cells write `model.pkl` and `vector.pkl`.

## Getting started

The notebook uses Python, pandas, NumPy, seaborn, Matplotlib, NLTK, and scikit-learn. No dependency manifest or pinned environment is included.

1. Open `Fake_news_detection.ipynb` in Google Colab, which matches its Drive-mount cell.
2. Supply your own `train.csv` in the notebook's working directory. The code expects `id`, `title`, `author`, `text`, and `label` columns.
3. Run the cells in order. NLTK resource-download cells require network access.
4. Enter article text when the final input cell prompts for it.

For a local Jupyter environment with these dependencies already installed:

```bash
jupyter notebook Fake_news_detection.ipynb
```

Skip or adapt the `google.colab` Drive-mount cell for local use. Several exploration cells use fixed row indices and may need changes for a different dataset. The prediction helper interprets label `1` as fake and `0` as real; your labels must match that convention.

## Project structure

- `Fake_news_detection.ipynb` — training, evaluation, and prediction notebook.
- `Startup & Entrepreneurship.pptx` — accompanying presentation.

## Status and limits

This is an educational notebook, with no web UI or deployed prediction service. `train.csv`, trained pickle files, dataset provenance, and an automated test suite are not included. The notebook uses one classifier.

A classifier's prediction does not verify an article's truth.

## License

No license file is included. This project is for educational and research purposes. Use it responsibly and ethically.
