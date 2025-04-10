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

    # merge Title and Question
    df['combined_text'] = df['Title'].fillna('') + ' ' + df['Question'].fillna('')
    # spilt tags
    df['tag_list'] = df['Tags'].apply(lambda x: [t.strip() for t in x.split(',') if t.strip()])

    # how many times each 
    counter = Counter()
    for tags in df['tag_list']:
        counter.update(tags)

    # 只保留出现次数 >= min_freq 的标签
    # only save the one which have frequency >= min_freq
    valid_tags = set([t for t, freq in counter.items() if freq >= min_freq])
    df['tag_list'] = df['tag_list'].apply(lambda tags: [t for t in tags if t in valid_tags])
    df = df[df['tag_list'].map(len) > 0]

    return df

def build_and_train_model(df):
    X = df['combined_text']
    y_raw = df['tag_list']

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y_raw)

    # 拆成训练 / 测试集
    # split into train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 构建 Pipeline
    # build pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000)))
    ])

    # 训练
    # train
    pipeline.fit(X_train, y_train)

    return pipeline, mlb, X_test, y_test

if __name__ == "__main__":
    import time

    start_time = time.time()

    # 1. 加载并预处理
    # load and preprocess
    data_path = "data/stackoverflow_data.csv"  # 修改成你的数据路径
    df = load_and_preprocess_data(data_path, min_freq=2)

    # 2. 训练
    # train
    model, mlb, X_test, y_test = build_and_train_model(df)

    # 3. 评估
    # evaluate
    y_pred = model.predict(X_test)

    # 这里我们只计算 micro-average 的精确率、召回率、F1
    # here we only calculate micro-average precision, recall, F1
    p_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
    r_micro = recall_score(y_test, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)

    # 4. 结束时间
    #   end time
    end_time = time.time()
    time_taken = end_time - start_time


    print("\n✅ Evaluation Complete!")
    print(f"🕒 Time Taken: {time_taken:.2f} seconds")
    print(f"📌 Precision: {p_micro:.4f}")
    print(f"📌 Recall:    {r_micro:.4f}")
    print(f"📌 F1 Score:  {f1_micro:.4f}\n")

    # no need to do it now 再打印 Hamming Loss, classification_report
    #h_loss = hamming_loss(y_test, y_pred)
    #print(f"Hamming Loss: {h_loss:.4f}")
    #print("\n📊 Classification Report:\n")
    #print(classification_report(y_test, y_pred, zero_division=0))

    # 保存模型
    joblib.dump(model, "tagging_model.pkl")
    joblib.dump(mlb, "tagging_mlb.pkl")
    print("模型和标签编码器已保存。")
