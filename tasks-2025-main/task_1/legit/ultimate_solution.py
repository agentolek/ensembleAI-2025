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

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKEN = os.environ["ATHENA_TOKEN"]
URL = "http://149.156.182.9:6060/task-1/submit"
PRIVATE_DATASET_PATH = os.environ["PRIV_PATH"]
MIA_CKPT_PATH = os.environ["01_MIA_69_PATH"]

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
    
class PrivateDataset(TaskDataset):
    def __init__(self, victim_model, dataset):
        self.victim_model = victim_model
        self.dataset = dataset
        self.features = []
        self._prepare_data()

    def _prepare_data(self):
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        tensor_list = []
        self.victim_model.eval()
        for _, img, label in dataloader:
            img = img.to(DEVICE)
            with torch.no_grad():
                output = self.victim_model(img)
            temp = torch.cat((torch.flatten(img), torch.flatten(output), label.to(DEVICE).to(torch.float32)))
            tensor_list.append(temp)

        self.features = torch.stack(tensor_list, dim=0)


torch.serialization.add_safe_globals([MembershipDataset])
torch.serialization.add_safe_globals([PrivateDataset])

def inference_dataloader(dataset: MembershipDataset, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)

def load_tgt_model():
    victim_model = models.resnet18().to(DEVICE)
    victim_model.fc = torch.nn.Linear(512, 44).to(DEVICE)
    victim_model.load_state_dict(torch.load(MIA_CKPT_PATH, map_location=DEVICE))
    return victim_model

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

    model.eval()
    dataset = torch.load(PRIVATE_DATASET_PATH, map_location=DEVICE)
    tgt_model = load_tgt_model()
    priv_dataset = PrivateDataset(tgt_model, dataset)
    dataloader = DataLoader(dataset, 1)
    outputs_list = []

    for feature in dataloader:
        print(feature)
        feature = feature.to(DEVICE)

        with torch.no_grad():
            membership_output = model(feature)

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
