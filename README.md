# Fake News Classification with Machine Learning

This repository contains a complete, production-ready machine learning pipeline for detecting fake news. It is trained on the **WELFake Dataset** (72,134 news articles) using standard natural language processing (NLP) techniques and classic machine learning algorithms.

This project follows the structured, step-by-step educational coding style of **Việt Nguyễn AI**.

---

## 📊 Model Performance

We trained and evaluated two models on an 80/20 train/test split:

| Model | Accuracy | F1-Score (Real) | F1-Score (Fake) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | `95.70%` | `0.96` | `0.96` |
| **Passive Aggressive Classifier** | **`95.84%`** | **`0.96`** | **`0.96`** |

*Note: The **Passive Aggressive Classifier** achieved the highest accuracy and is saved as the default inference model.*

---

## 📁 Project Structure

```bash
├── fake_new.py       # Main python script containing the training pipeline & demo test
├── best_model.pkl    # Pre-trained Passive Aggressive model
├── vectorizer.pkl    # Pre-trained TF-IDF vectorizer (10,000 features)
├── .gitignore        # Ignores the heavy CSV dataset and OS cache files
└── README.md         # Project documentation (this file)
```

> ⚠️ **Dataset note**: The raw dataset `fake_new.csv` (234MB) is ignored in git because of GitHub's 100MB file size limit. You can download the dataset directly from Kaggle: [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification).

---

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have python 3 installed. Install the required libraries:

```bash
pip install numpy pandas scikit-learn tqdm
```

### 2. Run the Pipeline
To run the full preprocessing, train both models, print evaluation reports, save the best model, and test predictions, run:

```bash
python fake_new.py
```

### 3. Usage & Inference
The script automatically exports the model weights. You can load and use the model in any python script:

```python
import pickle
import re

# Load assets
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# Inference
text = "BREAKING: Secret documents reveal that aliens have taken over the government."
cleaned = clean_text(text)
vec = vectorizer.transform([cleaned])
prediction = model.predict(vec)[0]

label = "FAKE" if prediction == 1 else "REAL"
print(f"Result: {label}")
```
