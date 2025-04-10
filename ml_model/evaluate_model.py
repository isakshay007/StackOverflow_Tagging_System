# evaluate_model.py

import time
import joblib
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import MultiLabelBinarizer

def load_test_data(csv_path, mlb_classes):
    df = pd.read_csv(csv_path)

    # 这里用 df['text'] 代替 'Title' + 'Question'
    # use df['text'] to substitute 'Title' + 'Question'
    df['combined_text'] = df['text'].fillna('')

    # 把 tags 列用 ',' 拆分 → 再过滤只保留模型见过的
    # split the tags column by ',' and filter only the tags seen by the model
    def filter_tags(x):
        raw = [t.strip() for t in x.split(',') if t.strip()]
        return [t for t in raw if t in mlb_classes]

    df['tag_list'] = df['tags'].apply(filter_tags)
    df = df[df['tag_list'].map(len) > 0]
    return df


if __name__ == "__main__":
    start_time = time.time()

    # 1. 加载已训练好的模型
    # load the trained model
    pipeline = joblib.load("tagging_model.pkl")
    mlb = joblib.load("tagging_mlb.pkl")

    # 2. 读取你的测试/验证数据
    # read your test/validation data
    test_data_path = "val_split.csv"  # 或者别的 CSV
    df_test = load_test_data(test_data_path, mlb.classes_)

    # 3. 构造 X,y
    # construct X,y
    X_test = df_test['combined_text']
    # 把标签转成 0/1
    y_test = mlb.transform(df_test['tag_list'])

    # 4. 模型预测
    # model prediction
    y_pred = pipeline.predict(X_test)

    # 5. 计算指标（micro-average）
    # calculate metrics (micro-average)
    p = precision_score(y_test, y_pred, average='micro', zero_division=0)
    r = recall_score(y_test, y_pred, average='micro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)

    elapsed = time.time() - start_time

    print("\n✅ Evaluation Complete!")
    print(f"🕒 Time Taken: {elapsed:.2f} seconds")
    print(f"📌 Precision: {p:.4f}")
    print(f"📌 Recall:    {r:.4f}")
    print(f"📌 F1 Score:  {f1:.4f}\n")
