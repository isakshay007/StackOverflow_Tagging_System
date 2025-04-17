#  StackOverflow Autonomous Tagging System

The StackOverflow Autonomous Tagging System is a solution designed to automatically generate relevant and high-quality tags for questions posted on StackOverflow. Accurate tagging plays a crucial role in content discoverability, expert routing, and information retrieval on community-driven platforms. However, manual tagging is often inconsistent, subjective, and error-prone.

This project uses a hybrid approach, combining both traditional machine learning techniques and modern deep learning models, to understand question semantics and predict tags with high precision.

---

##  Goals

- **Enhance Tag Discoverability**: Generate tags that accurately reflect question content to improve search visibility and content classification.
- **Minimize Manual Errors**: Reduce inconsistent or missing tags with an automated, consistent tagging pipeline.
- **Compare ML Models**: Evaluate traditional (HMM), classical (Logistic Regression), and deep learning (DistilBERT) approaches for tag prediction.
- **Scalable Architecture**: Build a modular system that supports future feedback loops and active learning extensions.

##  Models Used

- **Hidden Markov Model (HMM)** 
- **Logistic Regression** 
- **DistilBERT** 
---

##  Features

- **Multi-Model Inference**  
  Supports Logistic Regression, Hidden Markov Model (HMM), and DistilBERT for tag prediction.
- **Confidence-Based Tag Filtering**  
  Filters tags based on customizable probability thresholds to ensure relevance.
- **Interactive UI**  
  Built with Streamlit, allowing users to select models and get real-time tag predictions.
- **Model Benchmarking**  
  Compare traditional, classical, and deep learning models on accuracy, speed, and scalability.
- **Offline & Online Integration**  
  Supports both local inference and online model fetching via Hugging Face Hub.
---

##  Project Structure

```
StackOverflow_Tagging_System/
│
├── data/    
├── ml_models/               
├── models/             
├── notebooks/           
├── src/                
├── tests/              
├── requirements.txt    
└── README.md           
```

---


##  Setup & Installation

Follow the steps below to set up and run the StackOverflow Tagging System on your local machine.

### 1️ Clone the Repository

Start by cloning the GitHub repository to your local system:

```bash
git clone https://github.com/isakshay007/StackOverflow_Tagging_System.git
cd StackOverflow_Tagging_System

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model manually**
   ```bash
   python -m spacy download en_core_web_md
   # or use this direct link:
   # https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0.tar.gz
   ```

---

## Usage

### 🖥 Launch the Web App
```bash
streamlit run src/tag_ui_app.py
```

---

##  Models Implemented

| Model               | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| Hidden Markov Model| Custom implementation trained on emission and transition probabilities.     |
| Logistic Regression| TF-IDF based multi-label classifier using OneVsRest strategy.               |
| DistilBERT         | Lightweight Transformer fine-tuned for tag classification with confidence. |

---

##  Evaluation Metrics

Each model is assessed based on:

- **Precision@k**
- **Recall@k**
- **F1 Score**
- **Jaccard Similarity**
- **Exact Match Ratio**
- **Hamming Loss**
- **Training & Inference Time**


---

##  Dependencies

### Core Packages
- `streamlit`
- `pandas`
- `numpy<2.0`
- `joblib`
- `tqdm`

### Machine Learning
- `scikit-learn`
- `scipy`

### NLP & Transformers
- `spacy`
- `transformers`
- `torch`

### Hugging Face Integration
- `huggingface_hub`
- `hf_xet` (optional)

---

##  Troubleshooting

- **spaCy model not found**: Ensure you've downloaded `en_core_web_md`.
- **Missing `.pkl` files**: Run training scripts to generate them or download from HuggingFace.
- **Slow transformer loading**: Ensure models are cached or consider running on GPU.

---

## Demo

Check out the deployed Streamlit app here:  
 [StackOverflow Tagging System Demo](https://stackoverflow-tagging-system.streamlit.app/)

Explore how our models (Logistic Regression, HMM, and DistilBERT) automatically generate relevant tags for Stack Overflow-style questions in real-time!

## 👥 Contributors

- [Akshay Keerthi AS]
- [Taniskha Datar] 
- [Zimo Xing]

---

