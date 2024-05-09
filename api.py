from flask import Flask, request, jsonify
from predictor import predictor  # Import the instantiated predictor object
from voice_recognizer import recognize_speech_from_mic  # Import the voice recognizer function

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

@app.route('/recognize_voice', methods=['GET'])
def recognize_voice():
    # Recognize speech and convert it to text
    recognized_text = recognize_speech_from_mic()
    if recognized_text is None:
        return jsonify({"error": "Could not recognize speech"}), 500

    return jsonify({"recognized_text": recognized_text})

if __name__ == '__main__':
    app.run(debug=True)
