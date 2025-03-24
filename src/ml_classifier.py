"""
ml_classifier.py

此脚本用来加载训练好的模型，并对新输入文本（标题、描述、代码）进行标签预测。
"""

import joblib
import sys

def load_model_and_mlb(model_path="tagging_model.pkl", mlb_path="tagging_mlb.pkl"):
    model = joblib.load(model_path)
    mlb = joblib.load(mlb_path)
    return model, mlb

def predict_tags(model, mlb, title, description, code="", threshold=0.3):
    combined_text = title + " " + description + " " + code
    probs = model.predict_proba([combined_text])[0]  # 获取每个标签的概率
    predicted_indices = [i for i, p in enumerate(probs) if p >= threshold]
    predicted_labels = mlb.classes_[predicted_indices]
    return predicted_labels.tolist()

if __name__ == "__main__":
    # 获取模型路径参数（默认在当前目录）
    model_path = sys.argv[1] if len(sys.argv) > 1 else "tagging_model.pkl"
    mlb_path = sys.argv[2] if len(sys.argv) > 2 else "tagging_mlb.pkl"

    model, mlb = load_model_and_mlb(model_path, mlb_path)

    # 模拟一个新问题
    new_title = "Python string split and join"
    new_desc = "I want to split a string by spaces and then join it using hyphens."
    new_code = "s = 'hello world'; '-'.join(s.split())"

    predicted_tags = predict_tags(model, mlb, new_title, new_desc, new_code)
    print("预测的标签：", predicted_tags)
