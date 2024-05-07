import pandas as pd

#vectorize data
from sklearn.feature_extraction.text import TfidfVectorizer

from cleanData import clean_text

# Carregar os datasets
df = pd.read_csv('trainEN.csv')
df['clean_text'] = df['text'].apply(clean_text)
from sklearn.feature_extraction.text import TfidfVectorizer

#vectorizer = TfidfVectorizer(max_features=1000)  # Limitando a 1000 features para simplicidade
vectorizer = TfidfVectorizer(max_features=5000, min_df=3, max_df=0.7, ngram_range=(1, 2))

X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report

# Naive Bayes
#nb_classifier = MultinomialNB()
#nb_classifier.fit(X_train, y_train)
#nb_predictions = nb_classifier.predict(X_test)
#print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))

# SVM
from sklearn.model_selection import GridSearchCV

# Para SVM, por exemplo
#param_grid = {
#    'C': [0.1, 1, 10],
#    'gamma': ['scale', 'auto', 0.1, 1, 10],
#    'kernel': ['rbf', 'linear']
#}
#grid_search = GridSearchCV(SVC(), param_grid, cv=5)
#grid_search.fit(X_train, y_train)
#print("Melhores parâmetros: ", grid_search.best_params_)
#svm_predictions = grid_search.predict(X_test)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))

# Cross Validation VERY IMPORTANT
from sklearn.model_selection import cross_val_score, StratifiedKFold
skf = StratifiedKFold(n_splits=5)
scores = cross_val_score(svm_classifier, X, y, cv=skf)
print("Cross-validated scores:", scores)

# Passive Aggressive Classifier
pa_classifier = PassiveAggressiveClassifier(max_iter=50)
pa_classifier.fit(X_train, y_train)
pa_predictions = pa_classifier.predict(X_test)
print("Passive Aggressive Classifier Accuracy:", accuracy_score(y_test, pa_predictions))

#print("Naive Bayes Classification Report:\n", classification_report(y_test, nb_predictions))
print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))
print("Passive Aggressive Classification Report:\n", classification_report(y_test, pa_predictions))
