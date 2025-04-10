import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss
from hmm import HMM_Tagger

# === Load dataset ===
df = pd.read_csv("data/val_split.csv")  
test_df = df.sample(n=500, random_state=42).reset_index(drop=True)

# === Enhanced Evaluation ===
def evaluate_model(model, test_data, k=5, log_to_csv=False, csv_path="hmm_evaluation_log.csv"):
    print("Evaluating...")
    total = 0
    precision_sum = 0
    recall_sum = 0
    jaccard_sum = 0
    exact_match_count = 0

    all_true_tags = []
    all_pred_tags = []
    log_rows = []

    for _, row in test_data.iterrows():
        sentence = row['text'] if isinstance(row['text'], str) else ""
        true_tags = set(t.strip() for t in row['tags'].split(',') if t.strip())
        predicted_tags = set(model.predict(sentence, top_n=k))

        true_positive = len(predicted_tags & true_tags)
        precision = true_positive / len(predicted_tags) if predicted_tags else 0
        recall = true_positive / len(true_tags) if true_tags else 0
        jaccard = len(predicted_tags & true_tags) / len(predicted_tags | true_tags) if predicted_tags | true_tags else 0
        exact_match = predicted_tags == true_tags

        if exact_match:
            exact_match_count += 1

        precision_sum += precision
        recall_sum += recall
        jaccard_sum += jaccard
        total += 1

        all_true_tags.append(list(true_tags))
        all_pred_tags.append(list(predicted_tags))

        if log_to_csv:
            log_rows.append({
                'text': sentence[:300] + '...' if len(sentence) > 300 else sentence,
                'true_tags': ', '.join(sorted(true_tags)),
                'predicted_tags': ', '.join(sorted(predicted_tags))
            })

    # === Metrics ===
    mlb = MultiLabelBinarizer()
    Y_true = mlb.fit_transform(all_true_tags)
    Y_pred = mlb.transform(all_pred_tags)

    micro_f1 = f1_score(Y_true, Y_pred, average='micro')
    macro_f1 = f1_score(Y_true, Y_pred, average='macro')
    h_loss = hamming_loss(Y_true, Y_pred)

    avg_precision = precision_sum / total
    avg_recall = recall_sum / total
    avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-10)
    avg_jaccard = jaccard_sum / total
    exact_match_accuracy = exact_match_count / total

    print("\n=== Evaluation Results ===")
    print(f"Precision@{k}:     {avg_precision:.4f}")
    print(f"Recall@{k}:        {avg_recall:.4f}")
    print(f"F1@{k}:            {avg_f1:.4f}")
    print(f"Jaccard@{k}:       {avg_jaccard:.4f}")
    print(f"Exact Match:       {exact_match_accuracy:.4f}")
    print(f"Hamming Loss:      {h_loss:.4f}")
    print(f"Micro F1 (all k):  {micro_f1:.4f}")
    print(f"Macro F1 (all k):  {macro_f1:.4f}")


    if log_to_csv:
        pd.DataFrame(log_rows).to_csv(csv_path, index=False)
        print(f"\nPrediction log saved to: {csv_path}")

# === Load model and run evaluation ===
hmm_tagger = HMM_Tagger()
hmm_tagger.load_model("hmm_model3.pkl")


evaluate_model(hmm_tagger, test_df, k=5, log_to_csv=True, csv_path="hmm_eval_log.csv")
