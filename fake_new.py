import re
import pickle
import numpy as np 
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
tqdm.pandas()

print(">>> 1. Loading dataset...")
df = pd.read_csv('fake_new.csv')
print(f"Dataset Shape: {df.shape}")
print("\nSample Data:")
print(df.head(3))
print("\nLabel counts:")
print(df['label'].value_counts())

# ----------------------------------------------------
# 2. DATA PREPROCESSING
# ----------------------------------------------------
print("\n>>> 2. Preprocessing data...")
# Handle missing values
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')

# Combine title and text to enrich context
df['title_text'] = df['title'] + " " + df['text']

# Define text cleaning function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove punctuation & special characters (keep letters, numbers, spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Cleaning text data (applying normalization & regex cleaning)...")
df['clean_title_text'] = df['title_text'].progress_apply(clean_text)

# ----------------------------------------------------
# 3. SPLIT TRAIN/TEST SETS
# ----------------------------------------------------
print("\n>>> 3. Splitting train and test sets...")
X = df['clean_title_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ----------------------------------------------------
# 4. TEXT VECTORIZATION (TF-IDF)
# ----------------------------------------------------
print("\n>>> 4. Extracting TF-IDF Features...")
# max_df=0.7: ignore terms that appear in more than 70% of documents
# max_features=10000: limit top features to keep execution fast and memory usage light
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=10000)

print("Fitting vectorizer on training data...")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------------------------------
# 5. MODEL TRAINING & EVALUATION
# ----------------------------------------------------
print("\n>>> 5. Training and evaluating models...")

# --- Model 1: Logistic Regression ---
print("\n--- Training Model 1: Logistic Regression ---")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_vec, y_train)

lr_pred = lr_model.predict(X_test_vec)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
print("Classification Report:")
print(classification_report(y_test, lr_pred, target_names=['Real', 'Fake']))

# --- Model 2: Passive Aggressive Classifier ---
print("\n--- Training Model 2: Passive Aggressive Classifier ---")
pac_model = PassiveAggressiveClassifier(max_iter=50, random_state=42)
pac_model.fit(X_train_vec, y_train)

pac_pred = pac_model.predict(X_test_vec)
pac_acc = accuracy_score(y_test, pac_pred)
print(f"Passive Aggressive Classifier Accuracy: {pac_acc:.4f}")
print("Classification Report:")
print(classification_report(y_test, pac_pred, target_names=['Real', 'Fake']))

# ----------------------------------------------------
# 6. SAVE THE BEST MODEL
# ----------------------------------------------------
print("\n>>> 6. Saving the best model...")
if pac_acc > lr_acc:
    best_model = pac_model
    best_acc = pac_acc
    model_name = "Passive Aggressive Classifier"
else:
    best_model = lr_model
    best_acc = lr_acc
    model_name = "Logistic Regression"

print(f"Saving {model_name} (Accuracy: {best_acc:.4f}) as the best model.")

# Save model and vectorizer
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Saved model to 'best_model.pkl' and vectorizer to 'vectorizer.pkl'.")

# ----------------------------------------------------
# 7. INTERACTIVE PREDICTION SYSTEM
# ----------------------------------------------------
print("\n>>> 7. Interactive Inference Test...")

def predict_news(news_text):
    # Preprocess the input text
    cleaned = clean_text(news_text)
    # Transform using saved vectorizer
    vec = vectorizer.transform([cleaned])
    # Predict using saved model
    pred = best_model.predict(vec)[0]
    prob = None
    
    # Try to get probability if supported (Logistic Regression supports predict_proba)
    if hasattr(best_model, "predict_proba"):
        prob = best_model.predict_proba(vec)[0]
        confidence = prob[pred]
    else:
        # Passive Aggressive doesn't support predict_proba natively,
        # but we can use decision_function for confidence score
        decision_score = best_model.decision_function(vec)[0]
        # map to pseudo probability via sigmoid
        confidence = 1 / (1 + np.exp(-abs(decision_score)))
        
    label = "FAKE" if pred == 1 else "REAL"
    return label, confidence

# Demo examples
demo_fake = "BREAKING: Secret documents reveal that aliens have taken over the government and are controlling the world's supply of chocolate."
demo_real = "The Federal Reserve kept its benchmark interest rate unchanged on Wednesday, pointing to solid economic growth and a strong job market."

for name, sample in [("Demo Fake News", demo_fake), ("Demo Real News", demo_real)]:
    label, conf = predict_news(sample)
    print(f"\nSample: {name}")
    print(f"Content: \"{sample[:100]}...\"")
    print(f"Prediction: {label} (Confidence: {conf:.2%})")

# Interactive prompt
print("\n" + "="*50)
print("Demo prediction setup complete. Ready to run.")
