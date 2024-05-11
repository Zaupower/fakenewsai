from flask import Flask, request, jsonify
#from predictor import predictor  # Import the instantiated predictor object
from socket_predictor import send_request
from twitter_evaluation import evaluate_tweets_of_month, get_tweets_by_month, get_latest_tweet

app = Flask(__name__)

@app.route('/check_string', methods=['POST'])
def check_string():
    data = request.get_json()
    if not data or 'string' not in data:
        return jsonify({"error": "No string provided"}), 400
    string_to_check = data['string']
    # Predict using the predictor object
    batch_request = {'predict_string': string_to_check}
    result = send_request(string_to_check)
    print("Single prediction response:", result)
    # Convert result to string
    result_str = "unreliable news" if result == 1 else "reliable news"
    return jsonify({"result": result_str})

@app.route('/evaluate_tweets', methods=['POST'])
def evaluate_tweets():
    data = request.json
    username = data.get('username')
    month = data.get('month')

    if not username or not month:
        return jsonify({"error": "Username and month are required"}), 400
    if not (1 <= month <= 12):
        return jsonify({"error": "Month must be between 1 and 12"}), 400
    month_tweets = get_tweets_by_month(username, month)
    batch_request = {'predict_batch': month_tweets}
    result = send_request(batch_request)
    print("Batch prediction response:", result)
    stats = evaluate_tweets_of_month(result)
    return jsonify(stats), 200

@app.route('/latest_tweet', methods=['POST'])
def latest_tweet():
    data = request.json
    username = data.get('username')

    if not username:
        return jsonify({"error": "Username is required"}), 400

    tweet = get_latest_tweet(username)
    return jsonify(tweet), 200

if __name__ == "__main__":
    app.run(debug=True)
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
