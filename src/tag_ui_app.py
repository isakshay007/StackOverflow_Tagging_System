import streamlit as st
import joblib
import pickle
import numpy as np
import torch
import types
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel
from huggingface_hub import hf_hub_download
from hmm import HMM_Tagger

# 🩹 Fix for PyTorch+Streamlit reload bug
if not hasattr(torch.classes, '__path__'):
    torch.classes.__path__ = types.SimpleNamespace(_path=[])

# ========= MODEL CLASS =========
class MiniTagTransformer(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

# ========= LOADERS =========
@st.cache_resource(show_spinner=False)
def load_ml():
    model = joblib.load("models/tagging_model.pkl")
    mlb = joblib.load("models/tagging_mlb.pkl")
    return model, mlb

@st.cache_resource(show_spinner=False)
def load_hmm():
    model = HMM_Tagger()
    model.load_model("models/hmm_model3.pkl")
    return model

@st.cache_resource(show_spinner=False)
def load_bert():
    model_path = hf_hub_download(repo_id="iakshay777/stackoverflow-tag-model", filename="trained_model.pt", repo_type="model")
    mlb_path = hf_hub_download(repo_id="iakshay777/stackoverflow-tag-model", filename="mlb.pkl", repo_type="model")
    with open(mlb_path, "rb") as f:
        mlb = pickle.load(f)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = MiniTagTransformer(num_tags=len(mlb.classes_))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model, mlb, tokenizer

def predict_ml(model, mlb, text, threshold=0.08):
    probs = model.predict_proba([text])[0]
    sorted_probs = sorted(zip(mlb.classes_, probs), key=lambda x: x[1], reverse=True)
    tags = [tag for tag, score in sorted_probs if score >= threshold]
    return tags[:5]

def predict_hmm(model, title, description):
    text = f"{title.strip()} {description.strip()}"
    return model.predict(text)[:5]

def predict_bert(text, model, tokenizer, mlb, threshold=0.05, show_top_k=5, fallback=True):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    top_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:show_top_k]
    indices = np.where(probs >= threshold)[0]
    tags = [mlb.classes_[i] for i in indices]
    if fallback and not tags:
        tags = [mlb.classes_[i] for i, _ in top_probs]
    return tags[:5]

# ========= STREAMLIT UI =========
st.set_page_config(page_title="StackOverflow Generator", layout="wide")

st.markdown(
    """
    <div style="display: flex; align-items: center;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" alt="StackOverflow Logo" width="40" style="margin-right: 10px;" />
        <h1 style="margin: 0;">StackOverflow Tag Generator</h1>
    </div>
    """,
    unsafe_allow_html=True
)

if "model_selected" not in st.session_state:
    st.session_state.model_selected = None

model_choice = st.selectbox("Choose a Tag Prediction Model below :", [
    "Logistic Regression (ML)",
    "Hidden Markov Model (HMM)",
    "DistilBERT Transformer"
], index=0)

if st.button("Select ✅ "):
    st.session_state.model_selected = model_choice
    if model_choice == "Logistic Regression (ML)":
        st.session_state.ml_model, st.session_state.mlb_ml = load_ml()
    elif model_choice == "Hidden Markov Model (HMM)":
        st.session_state.hmm_model = load_hmm()
    elif model_choice == "DistilBERT Transformer":
        st.session_state.bert_model, st.session_state.mlb_bert, st.session_state.tokenizer = load_bert()

# Input Section
if st.session_state.model_selected:
    st.subheader(f"📝 Enter Question for {st.session_state.model_selected}")
    title = st.text_input("Title", placeholder="e.g., How to resolve CORS error in JavaScript?")
    description = st.text_area(" Description", height=200, placeholder="Include details, errors, etc.")

    if st.button(" Generate Tags"):
        if not title.strip() and not description.strip():
            st.warning("Please provide at least a title or description.")
        else:
            with st.spinner("Generating tags..."):
                if st.session_state.model_selected == "Logistic Regression (ML)":
                    text = f"{title.strip()} {description.strip()}"
                    tags = predict_ml(st.session_state.ml_model, st.session_state.mlb_ml, text)
                elif st.session_state.model_selected == "Hidden Markov Model (HMM)":
                    tags = predict_hmm(st.session_state.hmm_model, title, description)
                elif st.session_state.model_selected == "DistilBERT Transformer":
                    text = f"{title.strip()} {description.strip()}"
                    tags = predict_bert(text, st.session_state.bert_model, st.session_state.tokenizer, st.session_state.mlb_bert)

                st.subheader("Predicted Tags:")
                st.write(", ".join(tags) if tags else "No tags found.")

# Footer
st.markdown("---")
st.caption("Built using Logistic Regression, HMM Model, and DistilBERT Transformer")
