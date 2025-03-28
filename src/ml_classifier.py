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

def predict_tags(model, mlb, title, description, code="", threshold=0.1):
    combined_text = title + " " + description + " " + code
    probs = model.predict_proba([combined_text])[0]  # 获取每个标签的概率
    predicted_indices = [i for i, p in enumerate(probs) if p >= threshold]
    predicted_labels = mlb.classes_[predicted_indices]
    return predicted_labels.tolist()

if __name__ == "__main__":
    # 模型路径参数（默认在当前目录）
    model_path = sys.argv[1] if len(sys.argv) > 1 else "../models/tagging_model.pkl"
    mlb_path = sys.argv[2] if len(sys.argv) > 2 else "../models/tagging_mlb.pkl"

    model, mlb = load_model_and_mlb(model_path, mlb_path)

    # 一组测试样例
    test_cases = [
        {
            "title": "How to use logistic regression for multi-label classification?",
            "desc": "Trying to build a multi-label classifier using scikit-learn. Should I use OneVsRestClassifier?",
            "code": "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression()\nmodel.fit(X, y)"
        },
        {
            "title": "Python string split and join",
            "desc": "I want to split a string by spaces and then join it using hyphens.",
            "code": "s = 'hello world'; '-'.join(s.split())"
        },
        {
            "title": "NullPointerException in Java",
            "desc": "I get a NullPointerException when trying to access a method from an object.",
            "code": "myObject.doSomething();"
        },
        {
            "title": "CSS flexbox center align",
            "desc": "How do I center a div using CSS flexbox?",
            "code": "display: flex; justify-content: center; align-items: center;"
        },
        {
            "title": "Merge two dictionaries in Python",
            "desc": "What is the most Pythonic way to merge two dictionaries?",
            "code": "merged = {**dict1, **dict2}"
        },
    ]

    # 执行测试并输出结果
    for i, case in enumerate(test_cases, 1):
        tags = predict_tags(model, mlb, case["title"], case["desc"], case["code"])
        print(f"\n【Test Case {i}】")
        print(f"title: {case['title']}")
        print(f"predicated tags: {tags}")
