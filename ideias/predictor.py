import socket
import pickle
import pandas as pd
import time
from helper_functions.clean_data import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

###
# Running instructions

# Start conda venv(base)
#> C:\Users\marce\anaconda3\condabin\activate.bat
# Start predictor
#> python .\predictor.py
###

class Predictor:
    def __init__(self):
        start_time = time.time()

        # Load and preprocess dataset
        self.df = pd.read_csv('datasets/trainEN.csv')
        # Combine title and text columns
        self.df['combined_text'] = self.df['title'] + " " + self.df['text']
        self.df['clean_text'] = self.df['combined_text'].apply(clean_text)

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
        print(f"Initialization time: {time.time() - start_time:.2f} seconds")

    def predict_string(self, input_string):
        # Preprocess and vectorize input string
        clean_input = clean_text(input_string)
        vectorized_input = self.vectorizer.transform([clean_input])

        # Make prediction
        prediction = self.svm_classifier.predict(vectorized_input)

        return prediction[0]

    def predict_batch(self, input_strings):
        # Preprocess and vectorize input strings
        clean_inputs = [clean_text(string) for string in input_strings]
        vectorized_inputs = self.vectorizer.transform(clean_inputs)

        # Make predictions
        predictions = self.svm_classifier.predict(vectorized_inputs)

        return predictions


def start_tcp_server(predictor, host='127.0.0.1', port=65432):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server started, listening on {host}:{port}")

        while True:
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                data = conn.recv(4096)
                if not data:
                    break
                
                # Deserialize the received datasets
                request = pickle.loads(data)
                print("Request: " + str(request))

                # Initialize result
                result = {"error": "Invalid request"}

                # Validate and process request
                if 'predict_string' in request:
                    if isinstance(request['predict_string'], str):
                        result = predictor.predict_string(request['predict_string'])
                    else:
                        result = {"error": "predict_string must be a valid string"}
                elif 'predict_batch' in request:
                    if isinstance(request['predict_batch'], list) and all(isinstance(item, str) for item in request['predict_batch']):
                        result = predictor.predict_batch(request['predict_batch'])
                    else:
                        result = {"error": "predict_batch must be a valid list of strings"}
                
                # Serialize the response
                print("result: "+str(result))
                response = pickle.dumps(result)
                
                conn.sendall(response)


if __name__ == "__main__":
    predictor = Predictor()
    start_tcp_server(predictor)

# Cross-validation
        # skf = StratifiedKFold(n_splits=5)
        # scores = cross_val_score(self.svm_classifier, X, y, cv=skf)
        # print("Cross-validated scores:", scores)
        # print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))


#start venv
# C:\Users\marce\anaconda3\condabin\activate.bat
#run predictor
#  python .\predictor.py
