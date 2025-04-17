from hmm import HMM_Tagger  # Adjust the import based on your file name
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load the HMM model
hmm_tagger = HMM_Tagger()
hmm_tagger.load_model("hmm_model.pkl")
print("Model Loaded")

# Predict raw tags for a new sentence
#sentence = "I wanted to share a set of configuration data between powershell, python and bash. " \
#"JSON is convenient for powershell and python, but bash does not have built-in support for JSON. " \
#"Most bash JSON examples found online don't work for JSON arrays nor Objects"

# Combine title and sentence into one combined string
title = "My project in full stack"
sentence = "Mean Stack in my web portfolio project using mongodb in the back and python flask in the front."

# Concatenate title and sentence to create a combined sentence
combined_sentence = title + " " + sentence

# Predict raw tags for the combined sentence
predicted_tags = hmm_tagger.predict(combined_sentence) 

# Preprocess the input sentence and raw predicted tags
def preprocess(text):
    return text.lower()

input_sentence = preprocess(sentence)

# Remove duplicate tags
predicted_tags = list(set([preprocess(tag) for tag in predicted_tags]))

# Use TF-IDF Vectorizer to convert text into numerical vectors
vectorizer = TfidfVectorizer()

# Combine the input sentence and raw tags into a single list for vectorization
all_text = [input_sentence] + predicted_tags

# Transform all text into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(all_text)

# Compute cosine similarities between the input sentence and each tag
cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# Set a threshold for similarity score (e.g., 0.2)
threshold = 0.1

# Filter out tags with similarity below the threshold
filtered_tags = [(tag, score) for tag, score in zip(predicted_tags, cosine_similarities) if score >= threshold]

# Sort tags by similarity score in descending order
sorted_tags = sorted(filtered_tags, key=lambda x: x[1], reverse=True)

# Print the top 5 tags
top_tags = sorted_tags[:5]
print("Top 5 Tags:")
for tag, score in top_tags:
    print(f"Tag: {tag}, Similarity Score: {score}")
