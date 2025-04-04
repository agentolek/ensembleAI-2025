import pandas as pd
import requests
import torch
import torch.nn as nn
import os
import torchvision.transforms as transforms

from dotenv import load_dotenv
from torch.utils.data import Dataset
from torchvision import models
from typing import Tuple
import torch.serialization

load_dotenv()

TOKEN = os.environ["ATHENA_TOKEN"]                      # Your token here
URL = "149.156.182.9:6060/task-1/submit"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
MEMBERSHIP_DATASET_PATH = "C:/Hackathons/ensembleAI-2025/tasks-2025-main/task_1/pub.pt"       # Path to priv_out_.pt
MIA_CKPT_PATH = "C:/Hackathons/ensembleAI-2025/tasks-2025-main/task_1/01_MIA_69.pt"                 # Path to 01_MIA_69.pt



allowed_models = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889])
])

class TaskDataset(Dataset):
    def __init__(self, transform=None):

        self.ids = []
        self.imgs = []
        self.labels = []

        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        return id_, img, label


torch.serialization.add_safe_globals([MembershipDataset])

def inference_dataloader(dataset: MembershipDataset, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)


def load_model(model_name, model_path):
    try:
        model: nn.Module = allowed_models[model_name](weights=None)
        model.fc = nn.Linear(model.fc.weight.shape[1], 44)
    except Exception as e:
        raise Exception(
            f"Invalid model class, {e}, only {allowed_models.keys()} are allowed"
        )

    try:
        model_state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(model_state_dict, strict=False)
        model.eval()
    except Exception as e:
        raise Exception(f"Invalid model, {e}")

    return model


def membership_prediction(model):
    dataset: MembershipDataset = torch.load(MEMBERSHIP_DATASET_PATH)
    dataloader = inference_dataloader(dataset, BATCH_SIZE)
    print(len(dataloader.dataset))
    outputs_list = []

    for _, img, _ in dataloader:
        img = img.to(DEVICE)

        with torch.no_grad():
            membership_output = model(img)

        outputs_list += membership_output.tolist()

    return pd.DataFrame(
        {
            "ids": dataset.ids,
            "score": outputs_list,
        }
    )


if __name__ == '__main__':
    model = models.resnet18()
    # model = load_model(model_name=..., model_path=...)                 # Insert model name and path to your model
    preds = membership_prediction(model)
    preds.to_csv("submission.csv", index=False)

    result = requests.post(
        URL,
        headers={"token": TOKEN},
        files={
            "csv_file": ("submission.csv", open("./submission.csv", "rb"))
        }
    )

    print(result.status_code, result.text)