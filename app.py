import streamlit as st
import pickle
import re
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector | Nhận diện Tin giả",
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

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spacing
    return text

# --- SIDEBAR & LANGUAGE SELECTOR ---
with st.sidebar:
    st.markdown("## 🌐 Ngôn ngữ / Language")
    language = st.selectbox(
        "Select Language / Chọn Ngôn ngữ",
        ["Tiếng Việt", "English"],
        label_visibility="collapsed"
    )
    st.markdown("---")

# --- TRANSLATIONS DICTIONARY ---
t = {
    "Tiếng Việt": {
        "title": "📰 Hệ thống Nhận diện Tin giả",
        "subtitle": "Xác thực tính trung thực của bất kỳ bài báo nào trong thời gian thực. Mô hình phân tích cấu trúc ngôn ngữ, từ vựng và sắc thái cú pháp.",
        "err_model": "⚠️ **Lỗi:** Không tìm thấy file mô hình (`best_model.pkl` và `vectorizer.pkl`). Vui lòng chạy lệnh `python fake_new.py` trước để huấn luyện mô hình.",
        "input_header": "### ✍️ Nhập Nội dung Bài viết",
        "title_label": "Tiêu đề bài báo / Headline",
        "title_placeholder": "Nhập tiêu đề bài viết vào đây...",
        "text_label": "Nội dung chi tiết bài báo",
        "text_placeholder": "Dán toàn bộ nội dung chi tiết bài viết vào đây...",
        "btn_analyze": "🔍 Phân tích Tính xác thực",
        "warning_empty": "⚠️ Vui lòng nhập ít nhất tiêu đề hoặc nội dung bài viết để phân tích.",
        "spinner": "Đang phân tích cấu trúc ngôn ngữ và đối chiếu đặc trưng...",
        "result_header": "### 📊 Kết quả phân tích",
        "fake_title": "🔴 PHÁT HIỆN TIN GIẢ (FAKE NEWS)",
        "fake_desc": "Cảnh báo! Cấu trúc câu, mật độ phân bố từ vựng và giọng điệu mang đậm tính giật gân, thêu dệt hoặc giật tít câu view (clickbait).",
        "real_title": "🟢 TIN THẬT XÁC THỰC (REAL NEWS)",
        "real_desc": "Bài báo hiển thị các đặc trưng thực tế cao, sử dụng từ vựng chuẩn mực, ngữ pháp khách quan và chuyên nghiệp tiêu chuẩn của báo chí chính thống.",
        "metric_label": "Kết luận",
        "metric_conf": "Độ tự tin của mô hình",
        "conf_bar": "**Mức độ tin cậy:**",
        "conf_note": "Chú thích: Độ tự tin được tính dựa trên khoảng cách của bài viết tới ranh giới quyết định (decision boundary) của mô hình máy học.",
        "side_header": "## ⚙️ Thông số Mô hình",
        "side_sub": "Khám phá các chỉ số của mô hình Máy học.",
        "side_model_type": "**Loại mô hình:**\nPassive Aggressive Classifier",
        "side_acc": "**Độ chính xác kiểm thử:**\n95.84%",
        "side_dataset_title": "### 📚 Thông tin Tập dữ liệu",
        "side_dataset_desc": "Mô hình được huấn luyện trên tập **WELFake Dataset** (72,134 bài báo tiếng Anh) từ các nguồn Kaggle, McIntire, Reuters và BuzzFeed Political.",
        "suggest_header": "💡 Nhấp để điền nhanh bài viết mẫu thử nghiệm:",
        "suggest_fake": "📰 Tin giả: Người ngoài hành tinh",
        "suggest_real_fed": "📰 Tin thật: Lãi suất Mỹ",
        "suggest_real_uk": "📰 Tin thật: Brexit Anh"
    },
    "English": {
        "title": "📰 Fake News Classifier",
        "subtitle": "Verify the credibility of any news article in real-time. Our model evaluates grammatical features, vocabulary, and semantic tone.",
        "err_model": "⚠️ **Error:** Model assets (`best_model.pkl` and `vectorizer.pkl`) not found. Please run `python fake_new.py` first to train and save the model.",
        "input_header": "### ✍️ Paste News Content",
        "title_label": "Article Title / Headline",
        "title_placeholder": "Enter the news headline here...",
        "text_label": "Article Body Text",
        "text_placeholder": "Paste the full article content here...",
        "btn_analyze": "🔍 Analyze Credibility",
        "warning_empty": "⚠️ Please enter at least a headline or article body text to analyze.",
        "spinner": "Analyzing linguistic patterns and cross-referencing features...",
        "result_header": "### 📊 Classification Result",
        "fake_title": "🔴 FAKE NEWS DETECTED",
        "fake_desc": "Caution! The linguistic patterns, vocabulary distribution, and tone are highly characteristic of sensationalized, fabricated, or clickbait content.",
        "real_title": "🟢 REAL NEWS VERIFIED",
        "real_desc": "This article displays high factual reporting characteristics, formal vocabulary, and objective syntax standard in credible journalism.",
        "metric_label": "Linguistic Match",
        "metric_conf": "Model Confidence Score",
        "conf_bar": "**Confidence Level:**",
        "conf_note": "Note: This confidence score is derived from the model's distance from the decision boundary. A higher score indicates a stronger match.",
        "side_header": "## ⚙️ Model Dashboard",
        "side_sub": "Explore the underlying Machine Learning model metrics.",
        "side_model_type": "**Model Type:**\nPassive Aggressive Classifier",
        "side_acc": "**Validation Accuracy:**\n95.84%",
        "side_dataset_title": "### 📚 WELFake Dataset Info",
        "side_dataset_desc": "Trained on **72,134** articles combining Kaggle, McIntire, Reuters, and BuzzFeed Political sources to analyze factual reporting patterns.",
        "side_author_title": "### 👨•💻 Author Style",
        "side_author_desc": "Created following the machine learning roadmap of **Việt Nguyễn AI**.",
        "suggest_header": "💡 Click to auto-fill a sample article for testing:",
        "suggest_fake": "📰 Fake: Aliens Conspiracy",
        "suggest_real_fed": "📰 Real: US Fed Rates",
        "suggest_real_uk": "📰 Real: UK Brexit"
    }
}

# Select translation
lang = t[language]

# --- SIDEBAR MAIN CONTENT ---
with st.sidebar:
    st.markdown(lang["side_header"])
    st.markdown(lang["side_sub"])
    
    st.info(lang["side_model_type"])
    st.success(lang["side_acc"])
    
    st.markdown("---")
    st.markdown(lang["side_dataset_title"])
    st.markdown(lang["side_dataset_desc"])
    
    st.markdown("---")
    st.markdown(lang["side_author_title"])
    st.markdown(lang["side_author_desc"])

# --- MAIN APP ---
st.title(lang["title"])
st.markdown(lang["subtitle"])

if not assets_loaded:
    st.error(lang["err_model"])
else:
    # --- SUGGESTIONS BUTTONS ---
    st.markdown(f"**{lang['suggest_header']}**")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button(lang["suggest_fake"], use_container_width=True):
            st.session_state.headline_val = "BREAKING: Secret documents reveal that aliens have taken over the government"
            st.session_state.body_val = "Secret leaked documents from the deep state show that extraterrestrial beings have infiltrated the highest levels of the federal government. According to anonymous sources, these aliens are currently controlling the world's financial systems and the global supply of chocolate. Several politicians have already been replaced by identical clones."
            st.rerun()
            
    with col_btn2:
        if st.button(lang["suggest_real_fed"], use_container_width=True):
            st.session_state.headline_val = "Federal Reserve keeps interest rates unchanged amid stable economic growth"
            st.session_state.body_val = "The Federal Reserve kept its benchmark interest rate unchanged on Wednesday, pointing to solid economic growth and a strong job market. The decision reflects the central bank's cautious approach to managing inflation while sustaining economic expansion. Fed Chairman Jerome Powell stated that the committee will continue to monitor incoming data before making adjustments."
            st.rerun()
            
    with col_btn3:
        if st.button(lang["suggest_real_uk"], use_container_width=True):
            st.session_state.headline_val = "UK Prime Minister makes new Brexit offer regarding European Union citizens"
            st.session_state.body_val = "The United Kingdom Prime Minister has presented a new Brexit proposal concerning the status of European Union citizens residing in the country. The offer aims to clarify residency rights and ensure reciprocal agreements for British citizens living in EU member states. European officials promised to study the offer carefully."
            st.rerun()

    st.markdown("---")
    st.markdown(lang["input_header"])
    
    title_input = st.text_input(
        lang["title_label"],
        value=st.session_state.get("headline_val", ""),
        placeholder=lang["title_placeholder"]
    )
    
    text_input = st.text_area(
        lang["text_label"],
        value=st.session_state.get("body_val", ""),
        placeholder=lang["text_placeholder"],
        height=200
    )
    
    # Analyze button
    if st.button(lang["btn_analyze"], use_container_width=True):
        if not title_input.strip() and not text_input.strip():
            st.warning(lang["warning_empty"])
        else:
            with st.spinner(lang["spinner"]):
                # Combine title and text
                full_text = title_input + " " + text_input
                
                # Preprocess text
                cleaned = clean_text(full_text)
                
                # Vectorize text
                vec = vectorizer.transform([cleaned])
                
                # Predict
                pred = best_model.predict(vec)[0]
                
                # Confidence score
                if hasattr(best_model, "predict_proba"):
                    prob = best_model.predict_proba(vec)[0]
                    confidence = prob[pred]
                else:
                    decision_score = best_model.decision_function(vec)[0]
                    confidence = 1 / (1 + np.exp(-abs(decision_score)))
                
                # Label mapping (0 = Real, 1 = Fake)
                label = "FAKE" if pred == 1 else "REAL"
                
                st.markdown("---")
                st.markdown(lang["result_header"])
                
                if label == "FAKE":
                    st.markdown(
                        f"""
                        <div style="background-color:#f8d7da; padding:20px; border-radius:10px; border-left:8px solid #dc3545; margin-bottom: 20px;">
                            <h3 style="color:#721c24; margin-top:0;">{lang["fake_title"]}</h3>
                            <p style="color:#721c24; font-size:16px; margin-bottom: 0;">
                                {lang["fake_desc"]}
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
                            <h3 style="color:#155724; margin-top:0;">{lang["real_title"]}</h3>
                            <p style="color:#155724; font-size:16px; margin-bottom: 0;">
                                {lang["real_desc"]}
                            </p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Display confidence metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label=lang["metric_label"], value="TIN GIẢ / FAKE" if label == "FAKE" else "TIN THẬT / REAL")
                with col2:
                    st.metric(label=lang["metric_conf"], value=f"{confidence:.2%}")
                
                st.markdown(lang["conf_bar"])
                st.progress(float(confidence))
                st.caption(lang["conf_note"])
