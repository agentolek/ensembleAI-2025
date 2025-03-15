import torch
import requests

if __name__ == "__main__":
    model_path = "randint50.onnx"
    weights_path = "model_weights.pth"

    url = "http://149.156.182.9:6060/task-4/submit"
    auth_token = "aHX36NuduY3bHcPVeIJMbuW5X0ZBUS"

    with open(onnx_model_path, "rb") as onnx_file, open(model_weights_path, "rb") as weights_file:
        headers = {
            'token': auth_token
        }
        files = {
            'file': (onnx_model_path, onnx_file, 'application/octet-stream'),
            'model_state_dict': (model_weights_path, weights_file, 'application/octet-stream')
        }
        response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        print("Model submitted successfully!")
        print(response.text)
    else:
        print(f"Failed to submit model. Status code: {response.status_code}")
        print(response.text)
