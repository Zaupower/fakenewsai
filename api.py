from flask import Flask, request, jsonify
from predictor import predictor  # Import the instantiated predictor object

app = Flask(__name__)

@app.route('/check_string', methods=['POST'])
def check_string():
    data = request.get_json()
    if not data or 'string' not in data:
        return jsonify({"error": "No string provided"}), 400

    string_to_check = data['string']

    # Predict using the predictor object
    result = predictor.predict_string(string_to_check)

    return jsonify({"result": bool(result)})

if __name__ == '__main__':
    app.run(debug=True)
