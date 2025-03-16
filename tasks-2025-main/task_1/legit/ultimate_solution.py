import pandas as pd
import torch
import torch.nn as nn
import os
import requests

from dotenv import load_dotenv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models
from typing import Tuple
from torcheval.metrics import BinaryAUROC

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKEN = os.environ["ATHENA_TOKEN"]
URL = "http://149.156.182.9:6060/task-1/submit"
PRIVATE_DATASET_PATH = os.environ["PRIV_PATH"]

# Model atakujący (RMIA)
class AttackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3117, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


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

def random_random():
    dataset: MembershipDataset = torch.load(PRIVATE_DATASET_PATH)
    outputs_list = []

    outputs_list = torch.rand(20000)

    return pd.DataFrame(
        {
            "ids": dataset.ids,
            "score": outputs_list,
        }
    )


def membership_prediction(model):

    if model is None: return random_random()

    dataset: MembershipDataset = torch.load(PRIVATE_DATASET_PATH, map_location=DEVICE)
    dataloader = inference_dataloader(dataset, 1)
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
    model = AttackModel().to(DEVICE)
    model.load_state_dict(torch.load("/net/tscratch/people/tutorial040/task1/task1_model.pt", map_location=DEVICE))
    model.eval()

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
