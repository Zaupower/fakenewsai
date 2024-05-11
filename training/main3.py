import pandas as pd
from clean_data import clean_text

# Carregar os dataset
df = pd.read_csv('../trainEN.csv')
df['clean_text'] = df['text'].apply(clean_text)

# Vectorizar dataset
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, min_df=3, max_df=0.7, ngram_range=(1, 2))

X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))

# Cross Validation
from sklearn.model_selection import cross_val_score, StratifiedKFold
skf = StratifiedKFold(n_splits=5)
scores = cross_val_score(svm_classifier, X, y, cv=skf)
print("Cross-validated scores:", scores)
print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))

# Test test dataset

# Load the test dataset
test_df = pd.read_csv('../test.csv')
test_df['clean_text'] = test_df['text'].apply(clean_text)

# Transform the test data using the same vectorizer that was fit on the training data
X_test_new = vectorizer.transform(test_df['clean_text'])
# Make predictions using the SVM classifier
test_predictions = svm_classifier.predict(X_test_new)
# Create a DataFrame with the test dataset IDs and the corresponding predictions
submit_df = pd.DataFrame({
    'id': test_df['id'],
    'label': test_predictions
})

# Save the DataFrame to a CSV file
submit_df.to_csv('submit.csv', index=False)

