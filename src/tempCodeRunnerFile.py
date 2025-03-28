best_path_prob = np.max(viterbi[:, -1])
        best_last_tag = np.argmax(viterbi[:, -1])

        best_path = [best_last_tag]
        for t in range(num_words - 1, 0, -1):
            best_path.insert(0, backpointer[best_path[0], t])

        predicted_tags = [self.tags[idx] for idx in best_path]
        print(f'Predicted Tags: {predicted_tags}')
        return predicted_tags