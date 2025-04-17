# evaluate_model.py

import time
import joblib
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    hamming_loss, jaccard_score
)
from sklearn.preprocessing import MultiLabelBinarizer

def load_test_data(csv_path, mlb_classes):
    df = pd.read_csv(csv_path)

    df['combined_text'] = df['text'].fillna('')


    def filter_tags(x):
        raw = [t.strip() for t in x.split(',') if t.strip()]
        return [t for t in raw if t in mlb_classes]

    df['tag_list'] = df['tags'].apply(filter_tags)
    df = df[df['tag_list'].map(len) > 0]
    return df

if __name__ == "__main__":
    start_time = time.time()

    pipeline = joblib.load("tagging_model.pkl")
    mlb = joblib.load("tagging_mlb.pkl")

    test_data_path = "val_split.csv"
    df_test = load_test_data(test_data_path, mlb.classes_)

    X_test = df_test['combined_text']
    y_test = mlb.transform(df_test['tag_list'])

    y_pred = pipeline.predict(X_test)

    precision = precision_score(y_test, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='micro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
    h_loss = hamming_loss(y_test, y_pred)
    jaccard = jaccard_score(y_test, y_pred, average='samples', zero_division=0)

    elapsed = time.time() - start_time

    print("\n Evaluation Complete!")
    print(f" Time Taken: {elapsed:.2f} seconds")
    print(f" Micro Precision: {precision:.4f}")
    print(f" Micro Recall:    {recall:.4f}")
    print(f" Micro F1 Score:  {f1:.4f}")
    print(f" Hamming Loss:    {h_loss:.4f}")
    print(f" Jaccard Score:   {jaccard:.4f}\n")
