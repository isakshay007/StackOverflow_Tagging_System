from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle

class MLClassifier:
    def __init__(self, model_type="random_forest"):
        if model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100)
        elif model_type == "svm":
            self.model = SVC(kernel="linear", probability=True)
        else:
            raise ValueError("Unsupported model type!")

    def train(self, X, y):
        """ Train ML classifier on feature matrix X and labels y """
        self.model.fit(X, y)

    def predict(self, X):
        """ Predict using trained classifier """
        return self.model.predict(X)

    def save_model(self, path):
        """ Save trained model """
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, path):
        """ Load trained model """
        with open(path, "rb") as f:
            self.model = pickle.load(f)

# Usage Example:
# classifier = MLClassifier(model_type="random_forest")
# classifier.train(X_train, y_train)
# predictions = classifier.predict(X_test)
