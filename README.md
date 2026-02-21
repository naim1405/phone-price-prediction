# Phone Price Prediction

A small ML inference project for predicting a phone price range category from device specifications.

This repository currently includes:
- a trained model file (`model.pkl`)
- a Gradio app (`app.py`) for running predictions from a UI
- a notebook (`model-training.ipynb`) used during model training
- dependency list (`requirements.txt`)

## Dataset

- Source: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
- The notebook downloads this dataset and uses `train.csv` for training.

## What this app does

`app.py` loads the pre-trained model from `model.pkl` and predicts a **price range class** from input features such as:
- battery power
- clock speed
- camera specs
- memory and RAM
- display resolution
- connectivity flags (3G/4G/WiFi/Bluetooth)

The output is shown as:
- `Predicted Price Range: <class>`

## Project structure

- `app.py` – Gradio inference interface
- `model.pkl` – serialized trained model used by the app
- `model-training.ipynb` – training workflow notebook
- `requirements.txt` – Python packages
- `mobile-price-classification.zip` – dataset/archive file in the repo

## Training process

The training workflow in `model-training.ipynb` is:

1. Download and load the dataset (`train.csv`).
2. Define features and target (`price_range`).
3. Build a preprocessing pipeline:
	- numeric: median imputation + standard scaling
	- categorical: most-frequent imputation + one-hot encoding
4. Split data into train and test sets (80/20).
5. Train and compare multiple models (Logistic Regression, SVM, KNN, Random Forest).
6. Select Logistic Regression, then run 5-fold cross-validation.
7. Tune hyperparameters with `GridSearchCV`.
8. Evaluate on test data (accuracy, macro F1, classification report, confusion matrix).
9. Save the final trained pipeline as `model.pkl`.

## Run locally

### 1) Create and activate a virtual environment (optional)

```bash
python3 -m venv env
source env/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Start the app

```bash
python app.py
```

Gradio will print a local URL in the terminal. Open it in your browser and provide the required phone specs.

## Notes

- `model.pkl` must be present in the project root (same level as `app.py`).
- The app predicts a class label (price range category), not an exact currency price.
