from flask import Flask, request, jsonify
import pandas as pd
from io import StringIO
import joblib
import sys
import os

# Get the current working directory
current_dir = os.getcwd()

# Adjust the path to the project root (assuming the notebook is two levels deep in the folder structure)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# Add the project root to the Python path
sys.path.append(project_root)
from helper_functions.clean_data import clean_text
app = Flask(__name__)

grid_search = joblib.load('clf.pkl')
pipeline = grid_search.best_estimator_

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        # Read CSV file
        data = pd.read_csv(file)
        
        # Check if necessary columns are present
        if not all(column in data.columns for column in ['id', 'title', 'author', 'text']):
            return jsonify({"error": "CSV must contain 'id', 'title', 'author', and 'text' columns"}), 400

        # Combine title and text columns
        data['combined_text'] = data['title'] + " " + data['text']

        # Clean the combined text
        data['clean_text'] = data['combined_text'].apply(clean_text)

        # Vectorize and predict
        X = pipeline.named_steps['tfidf'].transform(data['clean_text'])
        data['prediction'] = pipeline.named_steps['classifier'].predict(X)
        
        # Convert DataFrame back to CSV
        output = StringIO()
        data.to_csv(output, index=False)
        output.seek(0)
        
        return output.getvalue(), 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': 'attachment; filename=predictions.csv'
        }
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)