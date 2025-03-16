import os
import requests
import torch
from torchvision import models
from dotenv import load_dotenv


def test_model(path):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allowed_models = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }
    with open(path, "rb") as f:
        try:
            model: torch.nn.Module = allowed_models["resnet50"](weights=None)
            model.fc = torch.nn.Linear(model.fc.weight.shape[1], 10)
        except Exception as e:
            raise Exception(
                f"Invalid model class, {e=}, only {allowed_models.keys()} are allowed",
            )
        try:
            state_dict = torch.load(f, map_location=torch.device("cpu"))
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            out = model(torch.randn(1, 3, 32, 32))
        except Exception as e:
            raise Exception(f"Invalid model, {e=}")
        assert out.shape == (1, 10), "Invalid output shape"


if __name__ == '__main__':
    load_dotenv()
    TOKEN = os.getenv('API_TOKEN')
    URL = "http://149.156.182.9:6060/task-3/submit"
    MODEL_PATH = "randint50_55_1.8.pt"

    # model = models.resnet50(weights=None)
    # model.fc = nn.Linear(model.fc.weight.shape[1], 10)
    # checkpoint = torch.load(MODEL_PATH)
    # model.load_state_dict(checkpoint)

    test_model(MODEL_PATH)

    response = requests.post(
        URL,
        headers={
            "token": TOKEN,
            "model-name": "resnet50"
        },
        files={
            "model_state_dict": open(MODEL_PATH, "rb")
        }
    )
    print(response.status_code, response.text)