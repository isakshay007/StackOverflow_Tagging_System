import numpy as np

class CustomHMM:
    def __init__(self, n_states, n_observations, n_iter=50):
        """
        Implements a Hidden Markov Model (HMM) from scratch.
        - `n_states`: Number of hidden states (tags)
        - `n_observations`: Number of unique words in the vocabulary
        - `n_iter`: Number of training iterations
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self.n_iter = n_iter

        # Initialize transition, emission, and start probabilities randomly
        self.transition_probs = np.random.rand(n_states, n_states)
        self.transition_probs /= self.transition_probs.sum(axis=1, keepdims=True)

        self.emission_probs = np.random.rand(n_states, n_observations)
        self.emission_probs /= self.emission_probs.sum(axis=1, keepdims=True)

        self.start_probs = np.random.rand(n_states)
        self.start_probs /= self.start_probs.sum()

    def forward(self, observations):
        """Computes observation probability using Forward Algorithm."""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))

        alpha[0] = self.start_probs * self.emission_probs[:, observations[0]]

        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = self.emission_probs[j, observations[t]] * np.sum(alpha[t - 1] * self.transition_probs[:, j])

        return alpha

    def viterbi(self, observations):
        """Finds most likely sequence of hidden states using Viterbi Algorithm."""
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        delta[0] = self.start_probs * self.emission_probs[:, observations[0]]

        for t in range(1, T):
            for j in range(self.n_states):
                max_prob = delta[t - 1] * self.transition_probs[:, j]
                psi[t, j] = np.argmax(max_prob)
                delta[t, j] = max_prob[psi[t, j]] * self.emission_probs[j, observations[t]]

        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def train(self, X, y):
        """Trains HMM using Expectation-Maximization (Baum-Welch)."""
        for iteration in range(self.n_iter):
            new_transition_probs = np.zeros_like(self.transition_probs)
            new_emission_probs = np.zeros_like(self.emission_probs)

            for i, sequence in enumerate(X):
                tag_sequence = y[i]
                for t in range(len(sequence) - 1):
                    new_transition_probs[tag_sequence[t], tag_sequence[t + 1]] += 1
                    new_emission_probs[tag_sequence[t], sequence[t]] += 1

            self.transition_probs = new_transition_probs / new_transition_probs.sum(axis=1, keepdims=True)
            self.emission_probs = new_emission_probs / new_emission_probs.sum(axis=1, keepdims=True)

            print(f"Iteration {iteration + 1}/{self.n_iter} complete.")

        print("✅ HMM training completed!")
