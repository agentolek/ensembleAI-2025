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

TOKEN = os.environ["ATHENA_TOKEN"]
URL = "149.156.182.9:6060/task-1/submit"
PRIVATE_DATASET_PATH = "tasks-2025-main/task_1/priv_out.pt"

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

def membership_prediction():

    dataset: MembershipDataset = torch.load(PRIVATE_DATASET_PATH)
    outputs_list = []

    outputs_list = torch.rand(20000)

    return pd.DataFrame(
        {
            "ids": dataset.ids,
            "score": outputs_list,
        }
    )


if __name__ == '__main__':
    preds = membership_prediction()
    preds.to_csv("submission.csv", index=False)

    result = requests.post(
        URL,
        headers={"token": TOKEN},
        files={
            "csv_file": ("submission.csv", open("./submission.csv", "rb"))
        }
    )

    print(result.status_code, result.text)
