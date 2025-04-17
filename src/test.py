from hmm import HMM_Tagger  # Adjust the import based on your file name
import numpy as np

# Load the HMM model
hmm_tagger = HMM_Tagger()
hmm_tagger.load_model("hmm_model3.pkl")
print("Model Loaded")

# Predict raw tags for a new sentence
# sentence = "I wanted to share a set of configuration data between powershell, python and bash. " \
# "JSON is convenient for powershell and python, but bash does not have built-in support for JSON. " \
# "Most bash JSON examples found online don't work for JSON arrays nor Objects"

# Combine title and sentence into one combined string
# title = "My project in full stack"
# sentence = "Mean Stack in my web portfolio project using mongodb in the back and python flask in the front."

# Concatenate title and sentence to create a combined sentence
# combined_sentence = "My project in full stack.Mean Stack in my web portfolio project using mongodb in the back and python flask in the front."

example_question_1 = """
How to resolve CORS error when making a fetch request in JavaScript?
 
I'm working on a front-end application using plain JavaScript, and I'm trying to make a fetch call to an external API (https://api.example.com/data). However, I'm getting the following error in the console:
 
Access to fetch at 'https://api.example.com/data' from origin 'http://localhost:3000' has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present on the requested resource.
 
I've tried setting mode: 'no-cors' in the fetch request, but it doesn't help. I'm not using any backend, just a simple HTML file served locally. Is there any way to fix this issue on the front-end or do I need to configure something on the server?
 
Here's my fetch code:
 
fetch('https://api.example.com/data', {
  method: 'GET',
  headers: {
    'Content-Type': 'application/json'
  }
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error(error));
"""

# Predict raw tags for the combined sentence

tags = hmm_tagger.predict_with_viterbi(example_question_1)
print("Predicted Tags:", tags)

# hmm_tagger.predict1(example_question_1) 
# Preprocess the input sentence and raw predicted tags
def preprocess(text):
    return text.lower()

input_sentence = preprocess(example_question_1)





