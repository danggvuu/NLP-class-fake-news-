# Phân loại tin tức giả mạo bằng Machine Learning (WELFake Dataset)

[English Version Below](#english-version)

Dự án này xây dựng một quy trình máy học (Machine Learning Pipeline) hoàn chỉnh để nhận diện tin tức giả mạo (Fake News). Mô hình được huấn luyện trên tập dữ liệu **WELFake Dataset** (gồm 72,134 bài báo) sử dụng các kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) cơ bản kết hợp với thuật toán học máy cổ điển.

Dự án tuân theo phong cách lập trình cấu trúc, tường tận của kênh đào tạo **Việt Nguyễn AI**.

---

## 📊 Hiệu suất Mô hình

Chúng tôi đã huấn luyện và đánh giá hai mô hình trên tập kiểm thử (tỷ lệ phân chia tập train/test là 80/20):

| Mô hình | Độ chính xác (Accuracy) | F1-Score (Tin thật - Real) | F1-Score (Tin giả - Fake) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | `95.70%` | `0.96` | `0.96` |
| **Passive Aggressive Classifier** | **`95.84%`** | **`0.96`** | **`0.96`** |

*Lưu ý: Mô hình **Passive Aggressive Classifier** đạt độ chính xác cao nhất và đã được xuất ra làm mô hình dự đoán chính cho ứng dụng Web.*

---

## 📁 Cấu trúc Thư mục

```bash
├── fake_new.py       # File script Python huấn luyện pipeline & demo thử nghiệm
├── app.py            # Code ứng dụng giao diện web trực quan (Streamlit App)
├── requirements.txt  # Danh sách thư viện cần thiết để deploy web app lên cloud
├── best_model.pkl    # Trọng số mô hình tốt nhất đã được lưu (Passive Aggressive)
├── vectorizer.pkl    # Bộ trích xuất đặc trưng TF-IDF vectorizer (10,000 features)
├── .gitignore        # Cấu hình lọc bỏ qua file dataset nặng và cache của Python/OS
└── README.md         # Tài liệu hướng dẫn dự án (file này)
```

> ⚠️ **Lưu ý về Dataset**: File dữ liệu gốc `fake_new.csv` (234MB) được lọc bỏ qua trong git vì giới hạn kích thước file 100MB của GitHub. Bạn có thể tải trực tiếp tập dữ liệu tại Kaggle: [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification).

---

## 🚀 Hướng dẫn Chạy local

### 1. Cài đặt Thư viện
Hãy đảm bảo bạn đã cài đặt Python 3 trên máy. Cài đặt các thư viện phụ thuộc bằng lệnh:

```bash
pip install -r requirements.txt
```

### 2. Huấn luyện lại Pipeline Mô hình
Để tự động chạy tiền xử lý dữ liệu, huấn luyện cả 2 mô hình, in báo cáo chi tiết và lưu mô hình tốt nhất, chạy:

```bash
python fake_new.py
```

### 3. Khởi động Web App ở Local
Để chạy thử ứng dụng Web có giao diện nhập liệu trực quan ngay tại máy tính của bạn, dùng lệnh:

```bash
streamlit run app.py
```

---

## 🌐 Triển khai lên Web (Nhận Link truy cập Miễn phí)

Bạn có thể deploy ứng dụng này lên cloud hoàn toàn miễn phí qua cổng **Streamlit Community Cloud** để chia sẻ link cho người khác dùng thử.

### Các bước thực hiện:
1. Đẩy toàn bộ thay đổi mới nhất của dự án lên GitHub của bạn (nhớ commit các file `app.py`, `requirements.txt`, `best_model.pkl`, và `vectorizer.pkl`).
2. Truy cập trang web **[Streamlit Community Cloud](https://share.streamlit.io/)** và đăng nhập bằng chính tài khoản GitHub của bạn.
3. Chọn **New app** (hoặc **Deploy an app**).
4. Điền các thông số liên kết:
   - **Repository**: Chọn đúng repo của bạn (ví dụ: `danggvuu/NLP-class-fake-news-`).
   - **Branch**: Nhập `main`.
   - **Main file path**: Nhập `app.py`.
5. Bấm nút **Deploy!**.

Sau khoảng 1-2 phút thiết lập môi trường, ứng dụng Web sẽ hiển thị trực tuyến. Bạn sẽ nhận được link công khai dạng `https://ten-cua-ban.streamlit.app/` để sử dụng!

---

<a id="english-version"></a>

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
