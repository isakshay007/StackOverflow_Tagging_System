### ml_classifier.py

import joblib
import sys

def load_model_and_mlb(model_path="tagging_model.pkl", mlb_path="tagging_mlb.pkl"):
    model = joblib.load(model_path)
    mlb = joblib.load(mlb_path)
    return model, mlb

def predict_tags(model, mlb, title, description, threshold=0.1):
    combined_text = title + " " + description
    probs = model.predict_proba([combined_text])[0]

    prob_dict = dict(zip(mlb.classes_, probs))
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    predicted_labels = [label for label, score in sorted_probs if score >= threshold]
    return predicted_labels, sorted_probs

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "tagging_model.pkl"
    mlb_path = sys.argv[2] if len(sys.argv) > 2 else "tagging_mlb.pkl"

    model, mlb = load_model_and_mlb(model_path, mlb_path)

    test_cases = [
        {
            "title": "connect sql to visual studio code",
            "desc": "Good afternoon, I am trying to connect my SQL database to Visual Studio Code..."
        },
        {
            "title": "Outlook Automation from Access 'exhausting all methods'",
            "desc": "I am using VBA in Access to automate emails through Outlook..."
        },
        {
            "title": "How to deploy a React app on Firebase",
            "desc": "I want to host my ReactJS web app using Firebase Hosting, but I'm having trouble configuring the deployment..."
        },
        {
            "title": "How to create custom widgets in Flutter",
            "desc": "I'm building a mobile app with Flutter and want to create reusable custom widgets with Dart..."
        },
        {
            "title": "Why does my Tailwind CSS style not apply inside a React component?",
            "desc": "I'm building a frontend with React and Tailwind CSS, but some classes like text-center and bg-red-500 are not applying..."
        },
    ]

    for i, case in enumerate(test_cases, 1):
        tags, scores = predict_tags(model, mlb, case["title"], case["desc"])
        print(f"\n【Test Case {i}】")
        print(f"title: {case['title']}")
        print(f"predicted tags: {tags}")
        print("Top tag probabilities:")
        for tag, score in scores[:5]:
            print(f"  {tag}: {score:.3f}")