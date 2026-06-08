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
├── app.py            # Streamlit web application code
├── requirements.txt  # Python package dependencies for web deployment
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
pip install -r requirements.txt
```

### 2. Run the Pipeline
To run the full preprocessing, train both models, print evaluation reports, save the best model, and test predictions, run:

```bash
python fake_new.py
```

### 3. Run the Web App Locally
To launch the interactive Streamlit web application on your local machine, run:

```bash
streamlit run app.py
```

---

## 🌐 Cloud Deployment (Get a Free Public Link)

You can deploy this application for free on **Streamlit Community Cloud** so anyone can use it via a browser link.

### Step-by-Step Deployment:
1. Push all your changes to GitHub (ensure `app.py`, `requirements.txt`, `best_model.pkl`, and `vectorizer.pkl` are committed).
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and log in using your GitHub account.
3. Click **New app** (or **Deploy an app**).
4. Fill in the deployment details:
   - **Repository**: Choose `danggvuu/NLP-class-fake-news-` (or your repo name).
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy!**

Within 1-2 minutes, your web application will be live, and you will receive a public link (e.g., `https://your-app-name.streamlit.app/`) that you can add to your GitHub description or share with others!
