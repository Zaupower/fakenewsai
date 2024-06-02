from flask import Flask, request, jsonify
import pandas as pd
from io import StringIO
import joblib
import sys
import os
from clean_data import clean_text

# Get the current working directory
current_dir = os.getcwd()

# Adjust the path to the project root (assuming the notebook is two levels deep in the folder structure)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# Add the project root to the Python path
sys.path.append(project_root)

app = Flask(__name__)

model = joblib.load('text_classification_pipeline.pkl')

@app.route('/predict_csv', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        input_data = pd.read_csv(file)
        
        # Combine title and text columns and clean the text
        input_data['clean_text'] = input_data['text'].apply(clean_text)
        
        # Make predictions
        predictions = model.predict(input_data['clean_text'])
        
        # Create a new DataFrame with the ID and predicted labels
        results = pd.DataFrame({'id': input_data['id'], 'label': predictions})
        
        output = StringIO()
        results.to_csv(output, index=False)
        output.seek(0)
        
        return output.getvalue(), 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': 'attachment; filename=predictions.csv'
        }
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/predict_string', methods=['POST'])
def predict_string():
    data = request.json
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    try:
        input_text = data['text']
        
        # Clean the text
        clean_input_text = clean_text(input_text)
        
        # Make prediction
        prediction = model.predict([clean_input_text])
        
        # Convert numpy int64 to Python int
        prediction = int(prediction[0])
        
        # Return the prediction result
        #1: unreliable
        #0: reliable
        return jsonify({"prediction": prediction}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)