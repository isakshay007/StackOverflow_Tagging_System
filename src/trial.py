from hmm import HMMTagger  # Adjust the import based on your file name

# Load the model
hmm_tagger = HMMTagger()
hmm_tagger.load_model("hmm_model.pkl")
print("model Loaded")
# Predict tags for a new sentence
sentence = "I wanted to share a set of configuration data between powershell, python and bash." \
" JSON is convenient for powershell and python, but bash does not have built-in support for JSON. " \
"Most bash JSON examples found online don't work for JSON arrays nor Objects"
predicted_tags = hmm_tagger.predict(sentence)
print(f'Predicted Tags: {predicted_tags}')



