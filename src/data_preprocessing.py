import pandas as pd
import re

def clean_text(text):
    """ Remove special characters, convert to lowercase """
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower()

def load_data(filepath):
    """ Load data from CSV or JSON """
    return pd.read_csv(filepath)

# Usage Example:
# df = load_data("stackoverflow_questions.csv")
# df["cleaned_text"] = df["question_text"].apply(clean_text)
