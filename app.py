import streamlit as st
import joblib
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

# ===============================
# LOAD ML MODEL
# ===============================
ml_model = joblib.load("ml_model/ml_model.pkl")
tfidf = joblib.load("ml_model/tfidf_vectorizer.pkl")
# ===============================
# LOAD BERT / RoBERTa MODEL
# ===============================
MODEL_PATH = "bert_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
bert_model.eval()

# ===============================
# LOAD LLM (FOR EXPLANATION ONLY)
# ===============================
@st.cache_resource
def load_explainer():
    return pipeline(
        "text-generation",
        model="distilgpt2",
        max_new_tokens=60
    )

explainer = load_explainer()

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("📰 Fake News Detection System")

st.write(
    "This system predicts whether a news article is **Fake or Real** "
    "using Machine Learning and Deep Learning. "
    "An LLM is used to explain the prediction."
)

text = st.text_area("Enter News Text", height=200)

# ===============================
# PREDICTION FUNCTIONS
# ===============================
def ml_predict(text):
    vector = tfidf.transform([text])
    pred = ml_model.predict(vector)[0]
    conf = max(ml_model.predict_proba(vector)[0])
    return pred, conf


def bert_predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    conf = probs[0][pred].item()
    return pred, conf


def llm_explain(text, verdict):
    prompt = (
        f"The following news article was classified as {verdict}.\n"
        f"Explain in simple terms why this decision might have been made:\n\n"
        f"{text}\n\nExplanation:"
    )
    output = explainer(prompt)[0]["generated_text"]
    return output.split("Explanation:")[-1].strip()

# ===============================
# RUN ANALYSIS
# ===============================
if st.button("Analyze News"):

    if len(text.strip()) < 20:
        st.warning("Please enter a longer news article for reliable prediction.")
    else:
        ml_label, ml_conf = ml_predict(text)
        bert_label, bert_conf = bert_predict(text)

        # FINAL VERDICT → ML PRIORITY (SAFE FOR VIVA)
        if ml_label == 1:
            final_verdict = "Fake News"
        else:
            final_verdict = "Real News"

        # ===============================
        # DISPLAY RESULTS
        # ===============================
        st.subheader("🔍 Model Predictions")

        st.write(
            f"**ML Model:** "
            f"{'Fake' if ml_label == 1 else 'Real'} "
            f"(Confidence: {ml_conf:.2f})"
        )

        st.write(
            f"**BERT Model:** "
            f"{'Fake' if bert_label == 1 else 'Real'} "
            f"(Confidence: {bert_conf:.2f})"
        )

        st.markdown(f"## ✅ Final Verdict: **{final_verdict}**")

        st.caption(
            "Note: The ML model makes the final decision. "
            "BERT analyzes language realism and does not verify facts."
        )

        # ===============================
        # LLM EXPLANATION
        # ===============================
        st.subheader("🧠 LLM Explanation")

        with st.spinner("Generating explanation..."):
            explanation = llm_explain(text, final_verdict)

        st.write(explanation)
