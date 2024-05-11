import socket
import pickle

def send_request(request, host='127.0.0.1', port=65432):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(pickle.dumps(request))
        data = s.recv(4096)
        response = pickle.loads(data)
        return response

    # Example single string prediction


    # Example batch prediction
    batch_request = {'predict_batch': ['Example text for prediction', 'Another example text']}
    response = send_request(batch_request)
    print("Batch prediction response:", response)