import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import re
import pickle
# import spacy
# from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import dok_matrix, csr_matrix

# Load a pre-trained model (e.g., 'en_core_web_md' for medium-sized English model)
# nlp = spacy.load('en_core_web_md')

class HMM_Tagger:

    def __init__(self):
        print('Initializing HMMTagger...')
        self.tags = []                # List of all unique tags
        self.vocab = []               # List of all unique words
        self.tag_to_index = {}        # Map tag to index
        self.word_to_index = {}       # Map word to index

        self.pi = None                # Initial probabilities
        self.A = None                 # Transition probabilities
        self.B = None                 # Emission probabilities

    def save_model(self, file_path):
        model_data = {
            "pi": self.pi,
            "A": self.A,
            "B": self.B,
            "tags": self.tags,
            "word_to_index": self.word_to_index,
            "word_frequencies": self.word_frequencies 
        }
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f'Model saved successfully to {file_path}')

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.pi = model_data["pi"]
        self.A = model_data["A"]
        self.B = model_data["B"]
        self.tags = model_data["tags"]
        self.word_to_index = model_data["word_to_index"]
        self.word_frequencies = model_data.get("word_frequencies", {})
        
        print(f'Model loaded successfully from {file_path}')

    def preprocess(self, text):
        # print(f'Preprocessing text: {text[:50]}...')
        if not isinstance(text, str):
            text = ""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        words = text.split()
        # print(f'Preprocessed words: {words[:10]}')
        return words

    def prepare_data(self, data):
        print('Preparing data...')
        tag_sentences = defaultdict(list)
        vocab_set = set()

        for index, row in data.iterrows():
            tags = [t.strip() for t in row['tags'].split(',') if t.strip()]
            sentence = row['text'] if isinstance(row['text'], str) else ""
            words = self.preprocess(sentence)

            for tag in tags:
                tag_sentences[tag].append(words)

            vocab_set.update(words)

        self.tags = list(tag_sentences.keys())
        self.vocab = list(vocab_set)

        self.tag_to_index = {tag: i for i, tag in enumerate(self.tags)}
        self.word_to_index = {word: i for i, word in enumerate(self.vocab)}

        print(f'Total tags: {len(self.tags)}')
        print(f'Total words: {len(self.vocab)}')

        return tag_sentences

    def fit(self, data):
        print('Fitting the model to the data...')
        tag_sentences = self.prepare_data(data)
        self.word_frequencies = Counter()
        for tag, sentences in tag_sentences.items():
            for sentence in sentences:
                self.word_frequencies.update(sentence)
        self.train(tag_sentences)

    def train(self, tag_sentences):
        print('Training model...')
        num_tags = len(self.tags)
        num_words = len(self.vocab)

        print(f'Number of tags: {num_tags}, Number of words: {num_words}')

        self.pi = np.zeros(num_tags)
        self.A = dok_matrix((num_tags, num_tags))
        self.B = dok_matrix((num_tags, num_words))

        for tag, sentences in tag_sentences.items():
            tag_index = self.tag_to_index[tag]
            self.pi[tag_index] += len(sentences)

            for sentence in sentences:
                for i, word in enumerate(sentence):
                    if word in self.word_to_index:
                        word_index = self.word_to_index[word]
                        self.B[tag_index, word_index] += 1

                    if i > 0:  # Transition from previous word's tag to current tag
                        prev_tag_index = tag_index
                        self.A[prev_tag_index, tag_index] += 1

        print('Converting matrices to sparse format...')
        self.pi /= np.sum(self.pi)
        self.A = csr_matrix(self.A / np.maximum(self.A.sum(axis=1), 1))
        
        row_sums = np.array(self.B.sum(axis=1)).flatten()  # Get row sums as a flat array
        row_indices, col_indices = self.B.nonzero()  # Get indices of non-zero entries

        # Normalize non-zero entries
        for row, col in zip(row_indices, col_indices):
            if row_sums[row] > 0:
                self.B[row, col] /= row_sums[row]

        self.B = csr_matrix(self.B)  # Convert to csr_matrix for further operations

        print('Training completed successfully.')


    #not using this
    def predict1(self, sentence, top_n=5):
        print(f'Predicting tags for sentence: "{sentence}"')
        
        # Preprocess sentence
        words = self.preprocess(sentence)
        num_tags = len(self.tags)
        num_words = len(words)
        
        # Initialize Viterbi and backpointer matrices
        viterbi = np.zeros((num_tags, num_words))
        backpointer = np.zeros((num_tags, num_words), dtype=int)

        # Initialize the Viterbi matrix for the first word
        for tag_index in range(num_tags):
            word_index = self.word_to_index.get(words[0], -1)
            if word_index != -1:
                viterbi[tag_index, 0] = self.pi[tag_index] * self.B[tag_index, word_index]
            else:
                viterbi[tag_index, 0] = self.pi[tag_index] * (1e-6)

        # Iterate over the remaining words
        for t in range(1, num_words):
            word_index = self.word_to_index.get(words[t], -1)
            if word_index == -1:
                emission_probs = np.full(num_tags, 1e-6)
            else:
                emission_probs = np.array(self.B[:, word_index].todense()).flatten()
                emission_probs[emission_probs == 0] = 1e-6  # Handle zero probabilities

            transition_probs = self.A.multiply(viterbi[:, t - 1].reshape(-1, 1)).tocsc()
            probs = transition_probs * emission_probs

            best_prev_tags = np.argmax(probs, axis=0)
            best_probs = np.max(probs, axis=0)

            viterbi[:, t] = best_probs
            backpointer[:, t] = best_prev_tags

        best_path_prob = np.max(viterbi[:, -1])
        best_last_tag = np.argmax(viterbi[:, -1])

        best_path = [best_last_tag]
        for t in range(num_words - 1, 0, -1):
            best_path.insert(0, backpointer[best_path[0], t])

        predicted_tags = [self.tags[idx] for idx in best_path]

        # Split each predicted tag into individual tags (comma-separated)
        all_tags = []
        for tag in predicted_tags:
            all_tags.extend(tag.split(','))  # Split by commas and collect individual tags
        
        print(f'Raw Predicted Tags: {all_tags}')
        
        return all_tags  # Return raw predicted tags

    #Predict with only the emission probabilities
    def predict(self, sentence, top_n = 5):
        # print(f'Predicting tags for sentence: "{sentence}"')

        words = self.preprocess(sentence)
        num_tags = len(self.tags)

        tag_scores = np.log(self.pi + 1e-10)  # Avoid log(0)

        for word in words:
            word_index = self.word_to_index.get(word, -1)
            if word_index == -1:
                word_probs = np.log(np.full(num_tags, 1e-6))
            else:
                emission_probs = np.array(self.B[:, word_index].todense()).flatten()
                emission_probs[emission_probs == 0] = 1e-6
                word_probs = np.log(emission_probs)

            tag_scores += word_probs

        # Get top-N scored compound tags
        top_indices = np.argsort(tag_scores)[::-1][:top_n * 3]  # Fetch more in case of splitting
        tag_score_pairs = [(self.tags[i], tag_scores[i]) for i in top_indices]

        # Split and score each subtag
        subtag_scores = {}
        for compound_tag, score in tag_score_pairs:
            for subtag in compound_tag.split(','):
                subtag = subtag.strip()
                if subtag:  # Avoid empty entries
                    # Keep highest score if subtag appears multiple times
                    if subtag not in subtag_scores or subtag_scores[subtag] < score:
                        subtag_scores[subtag] = score

        # Sort individual subtags by score
        sorted_subtags = sorted(subtag_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top-N clean tags
        final_tags = [tag for tag, _ in sorted_subtags[:top_n]]
        print(f'Predicted Tags: {final_tags}')
        return final_tags

    #Predict with the Transition probabilities and the emission probabilities
    def predict_with_viterbi(self, sentence, top_n=10):
        import numpy as np
        from collections import Counter

        words = self.preprocess(sentence)
        num_tags = len(self.tags)
        num_words = len(words)

        if num_words == 0 or num_tags == 0:
            return []

        A_dense = self.A.toarray()
        B_dense = self.B.toarray()

        viterbi = np.zeros((num_tags, num_words))
        backpointer = np.zeros((num_tags, num_words), dtype=int)

        # Initialization
        for tag_index in range(num_tags):
            word_index = self.word_to_index.get(words[0], -1)
            emission = B_dense[tag_index, word_index] if word_index != -1 else 1e-6
            emission = emission if emission != 0 else 1e-6
            viterbi[tag_index, 0] = np.log(self.pi[tag_index] + 1e-10) + np.log(emission)

        # Recursion
        for t in range(1, num_words):
            word_index = self.word_to_index.get(words[t], -1)
            for curr_tag in range(num_tags):
                emission = B_dense[curr_tag, word_index] if word_index != -1 else 1e-6
                emission = emission if emission != 0 else 1e-6
                emission_log = np.log(emission)

                transitions = A_dense[:, curr_tag]
                transitions[transitions == 0] = 1e-6
                transition_log = np.log(transitions)
                scores = viterbi[:, t - 1] + transition_log + emission_log
                best_prev_tag = np.argmax(scores)
                max_prob = scores[best_prev_tag]


                viterbi[curr_tag, t] = max_prob
                backpointer[curr_tag, t] = best_prev_tag

        # Backtrack
        best_last_tag = np.argmax(viterbi[:, -1])
        best_path = [best_last_tag]
        for t in range(num_words - 1, 0, -1):
            best_path.insert(0, backpointer[best_path[0], t])

        # Collect predicted compound tags
        predicted_tags = [self.tags[i] for i in best_path]
        

        # Split compound tags
        tag_counter = Counter()
        for tag in predicted_tags:
            for subtag in tag.split(','):
                subtag = subtag.strip()
                if subtag:
                    tag_counter[subtag] += 1
        
        return [tag for tag, _ in tag_counter.most_common(top_n)]






# import pandas as pd
# training_dataframe = pd.read_csv("data/train_split.csv")
# tagger = HMM_Tagger()
# tagger.fit(training_dataframe)  # The one with Title, Question, Tags
# tagger.save_model('hmm_model3.pkl')  # Optional: save after retraining
