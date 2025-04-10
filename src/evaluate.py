# from sklearn.model_selection import train_test_split
import pandas as pd
from hmm import HMM_Tagger

# Load the dataset
test_df = pd.read_csv("data/val_split.csv")  

def evaluate_model(model, test_data, k=5):
    total = 0
    precision_sum = 0
    recall_sum = 0
    jaccard_sum = 0

    for _, row in test_data.iterrows():
        sentence = row['text'] if isinstance(row['text'], str) else ""
        true_tags = set([t.strip() for t in row['tags'].split(',') if t.strip()])
        predicted_tags = set(model.predict(sentence, top_n=k))

        true_positive = len(predicted_tags & true_tags)
        precision = true_positive / len(predicted_tags) if predicted_tags else 0
        recall = true_positive / len(true_tags) if true_tags else 0
        jaccard = len(predicted_tags & true_tags) / len(predicted_tags | true_tags) if predicted_tags | true_tags else 0

        precision_sum += precision
        recall_sum += recall
        jaccard_sum += jaccard
        total += 1

    avg_precision = precision_sum / total
    avg_recall = recall_sum / total
    avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-10)
    avg_jaccard = jaccard_sum / total

    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Recall@{k}:    {avg_recall:.4f}")
    print(f"F1@{k}:        {avg_f1:.4f}")
    print(f"Jaccard@{k}:   {avg_jaccard:.4f}")

# Load model and evaluate
hmm_tagger = HMM_Tagger()
hmm_tagger.load_model("hmm_model3.pkl")
evaluate_model(hmm_tagger, test_df, k=5)
