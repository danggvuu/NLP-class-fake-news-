import streamlit as st
import pickle
import re
import numpy as np

# Set page configuration with a premium look
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the saved model and vectorizer
@st.cache_resource
def load_assets():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    best_model, vectorizer = load_assets()
    assets_loaded = True
except FileNotFoundError:
    assets_loaded = False

# Text cleaning function matching the training phase
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spacing
    return text

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚙️ Model Dashboard")
    st.markdown("Explore the underlying Machine Learning model metrics.")
    
    # Model info cards
    st.info("**Model Type:**\nPassive Aggressive Classifier")
    st.success("**Validation Accuracy:**\n95.84%")
    
    st.markdown("---")
    st.markdown("### 📚 WELFake Dataset Info")
    st.markdown(
        "Trained on **72,134** articles combining Kaggle, McIntire, Reuters, and BuzzFeed Political sources. "
        "The model analyzes linguistic patterns to detect clickbait, propaganda, and factual reporting."
    )
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Author Style")
    st.markdown("Created following the machine learning roadmap of **Việt Nguyễn AI**.")

# --- MAIN APP ---
st.title("📰 Fake News Classifier")
st.markdown(
    "Verify the credibility of any news article in real-time. "
    "Our model evaluates grammatical features, vocabulary, and semantic tone."
)

if not assets_loaded:
    st.error(
        "⚠️ **Error:** Model assets (`best_model.pkl` and `vectorizer.pkl`) not found in the workspace. "
        "Please run `python fake_new.py` first to train and save the model."
    )
else:
    st.markdown("### ✍️ Paste News Content")
    
    title_input = st.text_input(
        "Article Title / Headline",
        placeholder="Enter the news headline here..."
    )
    
    text_input = st.text_area(
        "Article Body Text",
        placeholder="Paste the full article content here...",
        height=200
    )
    
    # Analyze button
    if st.button("🔍 Analyze Credibility", use_container_width=True):
        if not title_input.strip() and not text_input.strip():
            st.warning("⚠️ Please enter at least a headline or article body text to analyze.")
        else:
            with st.spinner("Analyzing linguistic patterns and cross-referencing features..."):
                # Combine title and text
                full_text = title_input + " " + text_input
                
                # Preprocess text
                cleaned = clean_text(full_text)
                
                # Vectorize text
                vec = vectorizer.transform([cleaned])
                
                # Predict
                pred = best_model.predict(vec)[0]
                
                # Confidence score (sigmoid on decision function)
                if hasattr(best_model, "predict_proba"):
                    prob = best_model.predict_proba(vec)[0]
                    confidence = prob[pred]
                else:
                    decision_score = best_model.decision_function(vec)[0]
                    confidence = 1 / (1 + np.exp(-abs(decision_score)))
                
                # Label mapping
                label = "FAKE" if pred == 1 else "REAL"
                
                st.markdown("---")
                st.markdown("### 📊 Classification Result")
                
                if label == "FAKE":
                    st.markdown(
                        f"""
                        <div style="background-color:#f8d7da; padding:20px; border-radius:10px; border-left:8px solid #dc3545; margin-bottom: 20px;">
                            <h3 style="color:#721c24; margin-top:0;">🔴 FAKE NEWS DETECTED</h3>
                            <p style="color:#721c24; font-size:16px; margin-bottom: 0;">
                                Caution! The linguistic patterns, vocabulary distribution, and tone are highly characteristic of sensationalized, fabricated, or clickbait content.
                            </p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.balloons()
                    st.markdown(
                        f"""
                        <div style="background-color:#d4edda; padding:20px; border-radius:10px; border-left:8px solid #28a745; margin-bottom: 20px;">
                            <h3 style="color:#155724; margin-top:0;">🟢 REAL NEWS VERIFIED</h3>
                            <p style="color:#155724; font-size:16px; margin-bottom: 0;">
                                This article displays high factual reporting characteristics, formal vocabulary, and objective syntax standard in credible journalism.
                            </p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Display confidence metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Linguistic Match", value=label)
                with col2:
                    st.metric(label="Model Confidence Score", value=f"{confidence:.2%}")
                
                st.markdown("**Confidence Level:**")
                st.progress(float(confidence))
                st.caption(
                    "Note: This confidence score is derived from the model's distance from the decision boundary. "
                    "A higher score indicates a stronger match to either real or fake news patterns."
                )
