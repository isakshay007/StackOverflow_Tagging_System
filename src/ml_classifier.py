# classifier.py

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score

# 从 model.py 导入我们封装的模型类
from model import MultiLabelModel

# ============ 示例：假设你的数据已经过清洗，只需做简单读写 ============
def load_preprocessed_data(csv_path):
    """
    假设 data 已经在其他脚本中清洗完毕，并存储为 csv。
    包含列：['combined_text', 'tags']。
    'combined_text' 是已清洗拼接好的文本和代码特征。
    'tags' 是标签列表或字符串(如 "['python','pandas']")。
    """
    df = pd.read_csv(csv_path)

    # 将字符串形式的列表转换为真正的列表 (如 "['python','pandas']" -> ['python','pandas'])
    import ast
    def parse_tags(tag_str):
        try:
            return ast.literal_eval(str(tag_str))
        except:
            return []

    df['tags'] = df['tags'].apply(parse_tags)
    return df

def main():
    # 1. 加载数据 (示例中指定了固定文件名)
    data_path = "data/stackoverflow_data.csv"  # 根据实际路径调整
    df = load_preprocessed_data(data_path)
    print("Data loaded. Shape:", df.shape)

    # 2. 标签多热编码
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['tags'])

    # 3. 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(
        df['combined_text'],
        y,
        test_size=0.2,
        random_state=42
    )

    # 4. 特征向量化 (TfidfVectorizer)
    vectorizer = TfidfVectorizer(max_features=5000,
                                 ngram_range=(1,2),
                                 stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # 5. 初始化并训练模型
    classifier = MultiLabelModel()  # 默认使用LogisticRegression
    classifier.train(X_train_vec, y_train)

    # 6. 模型评估
    y_pred = classifier.predict(X_test_vec)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=mlb.classes_))

    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Micro-F1: {micro_f1:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    # 7. 如果需要，对新问题进行预测
    new_text = "How to install numpy in python? pip install numpy - but I get errors."
    new_text_vec = vectorizer.transform([new_text])
    new_pred = classifier.predict(new_text_vec)

    # 把预测的多热编码转为实际标签
    new_tags = mlb.inverse_transform(new_pred)
    print(f"\nFor text: '{new_text}'\nPredicted tags: {new_tags[0]}")

    # 8. 保存模型(可选)
    # classifier.save("data/multilabel_model.pkl")
    # 也可把 vectorizer 和 mlb 一并保存，用于上线或后续推理

if __name__ == "__main__":
    main()
