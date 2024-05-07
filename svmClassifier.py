from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def vectorize(df, X, y):

    vectorizer = TfidfVectorizer(max_features=5000, min_df=3, max_df=0.7, ngram_range=(1, 2))

    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X, y
def svmClassifier(X_train, X_test, y_train):
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    svm_predictions = svm_classifier.predict(X_test)
    return svm_classifier, svm_predictions

def crossValidate(svm_classifier, X, y, y_test, svm_predictions):
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    skf = StratifiedKFold(n_splits=5)
    scores = cross_val_score(svm_classifier, X, y, cv=skf)
    return scores
