# Fake News Detection System (ML + Deep Learning)

## 📌 Project Overview
This project implements an **AI-based Fake News Detection System** that classifies news text as **Real or Fake** using both **Machine Learning (ML)** and **Deep Learning (DL)** approaches.

The system is designed to demonstrate:
- Classical ML text classification
- Transformer-based deep learning (BERT/RoBERTa)
- Practical deployment using a web interface

This project was developed as part of the **CS351 – Machine Learning / AI Semester Project**.

---

## 🎯 Objectives
- Detect fake vs real news using textual content
- Compare traditional ML models with deep learning models
- Build an interactive interface for live predictions
- Understand limitations of AI-based fake news detection

---

## 🧠 Models Used

### 1️⃣ Machine Learning Model
- **Algorithm:** Logistic Regression  
- **Feature Extraction:** TF-IDF Vectorization  
- **Purpose:**  
  Captures word frequency patterns commonly found in fake vs real news.

### 2️⃣ Deep Learning Model
- **Model:** BERT / RoBERTa (Transformer-based)
- **Framework:** HuggingFace Transformers + PyTorch
- **Purpose:**  
  Understands **semantic meaning and context** of news text.

---

## 📊 Dataset
- Source: Public Fake News datasets (True.csv & Fake.csv)
- Columns:
  - `text` → News article content
  - `label` → 0 = Real, 1 = Fake
- Dataset files are **excluded from GitHub** due to size limitations.
- Models were trained locally using the combined dataset.

---

## 🖥️ Web Interface
- Built using **Streamlit**
- Allows user to:
  - Enter any news text
  - Get predictions from ML and DL models
  - Compare results interactively

---

python -m streamlit run app.py
http://localhost:8501


## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
