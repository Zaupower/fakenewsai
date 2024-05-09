import pandas as pd
from cleanData import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


class Predictor:
    def __init__(self):
        # Load and preprocess dataset
        self.df = pd.read_csv('./data/trainEN.csv')
        self.df['clean_text'] = self.df['text'].apply(clean_text)

        # Vectorize dataset
        self.vectorizer = TfidfVectorizer(max_features=5000, min_df=3, max_df=0.7, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(self.df['clean_text'])
        y = self.df['label']

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train SVM model
        self.svm_classifier = SVC(kernel='linear', cache_size=4000)
        self.svm_classifier.fit(X_train, y_train)

        # Evaluate model
        svm_predictions = self.svm_classifier.predict(X_test)
        print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))
        print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))

        # Cross-validation
        skf = StratifiedKFold(n_splits=5)
        scores = cross_val_score(self.svm_classifier, X, y, cv=skf)
        print("Cross-validated scores:", scores)

    def predict_string(self, input_string):
        clean_input = clean_text(input_string)
        vectorized_input = self.vectorizer.transform([clean_input])
        prediction = self.svm_classifier.predict(vectorized_input)
        return prediction[0]


# Instantiate the Predictor class
predictor = Predictor()
