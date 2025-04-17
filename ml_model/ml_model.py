import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)
import joblib
from collections import Counter

def load_and_preprocess_data(data_path, min_freq=2):
    df = pd.read_csv(data_path, encoding="utf-8")

    if 'Title' not in df.columns or 'Question' not in df.columns or 'Tags' not in df.columns:
        raise ValueError("CSV 文件缺少必要的列: Title, Question, Tags")

    df['combined_text'] = df['Title'].fillna('') + ' ' + df['Question'].fillna('')

    df['tag_list'] = df['Tags'].apply(lambda x: [t.strip() for t in x.split(',') if t.strip()])

    counter = Counter()
    for tags in df['tag_list']:
        counter.update(tags)

    valid_tags = set([t for t, freq in counter.items() if freq >= min_freq])
    df['tag_list'] = df['tag_list'].apply(lambda tags: [t for t in tags if t in valid_tags])
    df = df[df['tag_list'].map(len) > 0]

    return df

def build_and_train_model(df):
    X = df['combined_text']
    y_raw = df['tag_list']

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000)))
    ])

    # train
    pipeline.fit(X_train, y_train)

    return pipeline, mlb, X_test, y_test

if __name__ == "__main__":
    import time

    start_time = time.time()

    data_path = "data/stackoverflow_data.csv"  
    df = load_and_preprocess_data(data_path, min_freq=2)

    model, mlb, X_test, y_test = build_and_train_model(df)

    y_pred = model.predict(X_test)

    # here we only calculate micro-average precision, recall, F1
    p_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
    r_micro = recall_score(y_test, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)


    end_time = time.time()
    time_taken = end_time - start_time


    print("\n Evaluation Complete!")
    print(f" Time Taken: {time_taken:.2f} seconds")
    print(f" Precision: {p_micro:.4f}")
    print(f" Recall:    {r_micro:.4f}")
    print(f" F1 Score:  {f1_micro:.4f}\n")

    joblib.dump(model, "tagging_model.pkl")
    joblib.dump(mlb, "tagging_mlb.pkl")

