import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from custom_hmm import CustomHMM  # Import the Custom HMM model

class StackOverflowHMMTagger:
    def __init__(self, n_states=10, n_iter=50):
        """Wrapper class for training and predicting StackOverflow tags using the custom HMM model."""
        self.vectorizer = CountVectorizer(max_features=7000)
        self.mlb = MultiLabelBinarizer()
        self.hmm_model = None
        self.n_states = n_states
        self.n_iter = n_iter

        os.makedirs("model", exist_ok=True)

    def load_preprocessed_data(self, file_path):
        """Loads preprocessed data and prepares feature vectors and labels."""
        print("\n📂 Loading Preprocessed Data...")
        df = pd.read_csv(file_path)
        X_text = df['Cleaned_Question'].astype(str) + " " + df['Cleaned_Title'].astype(str)
        df['Tags'] = df['Tags'].apply(lambda x: x.split(','))
        y = self.mlb.fit_transform(df['Tags'])

        return X_text, y

    def train(self, file_path):
        """Trains the custom HMM model using preprocessed data."""
        X_text, y = self.load_preprocessed_data(file_path)
        X = self.vectorizer.fit_transform(X_text).toarray()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.hmm_model = CustomHMM(n_states=self.n_states, n_observations=X.shape[1], n_iter=self.n_iter)
        print("\n⏳ Training Hidden Markov Model...")
        self.hmm_model.train(X_train, y_train)

        joblib.dump(self.hmm_model, "model/stackoverflow_hmm_tagger.pkl")
        joblib.dump(self.vectorizer, "model/count_vectorizer.pkl")
        joblib.dump(self.mlb, "model/tag_binarizer.pkl")
        print("💾 Model Saved Successfully!")

    def predict_tags(self, question_text):
        """Predicts relevant tags for a given StackOverflow question."""
        try:
            self.hmm_model = joblib.load("model/stackoverflow_hmm_tagger.pkl")
            self.vectorizer = joblib.load("model/count_vectorizer.pkl")
            self.mlb = joblib.load("model/tag_binarizer.pkl")

            vectorized_text = self.vectorizer.transform([question_text]).toarray()
            predicted_states = self.hmm_model.viterbi(vectorized_text[0])

            predicted_tags = self.mlb.classes_[predicted_states[:3]]
            return predicted_tags

        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return []

if __name__ == "__main__":
    tagger = StackOverflowHMMTagger()

    print("\n🚀 Starting Model Training...")
    tagger.train("data/stackoverflow_cleaned.csv")

    sample_question = "How to optimize SQL query performance?"
    predicted_tags = tagger.predict_tags(sample_question)
    print(f"\n🔮 Predicted Tags: {predicted_tags}")
