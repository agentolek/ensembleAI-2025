import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models
from typing import Tuple
from torcheval.metrics import BinaryAUROC


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
MEMBERSHIP_DATASET_PATH = "/net/tscratch/people/tutorial040/task1/pub.pt"
# MEMBERSHIP_DATASET_PATH = "tasks-2025-main/task_1/pub.pt"
BATCH_SIZE = 1


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
        self.membership = []

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]
    
torch.serialization.add_safe_globals([MembershipDataset])

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3073, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    

def train_loop(dataloader, model, loss_fn, optimizer):
    # size = len(dataloader.dataset)
    model.train()
    counter = 0

    for _, img, lbl, member in dataloader:
        
        img, lbl, member = img.to(DEVICE).to(torch.float32), lbl.to(DEVICE).to(torch.float32), member.to(DEVICE).to(torch.float32)

        X = torch.cat((img.flatten(), lbl))
        
        pred = model(X)
        loss = loss_fn(pred, member)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        counter += 1

        if counter % 100 == 0:
            loss, current = loss.item(), counter
            print(f"loss: {loss:>7f}  [{current:>5d}/{counter:>5d}]")


def test_loop(dataloader, model):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()

    test_preds = []
    correct_preds = dataloader.dataset.membership

    with torch.no_grad():

        for _, img, lbl, member in dataloader:
        
            img, lbl, member = img.to(DEVICE), lbl.to(DEVICE), member.to(DEVICE)

            X = torch.cat((img.flatten(), lbl))
            pred = model(X)
            test_preds.append(pred)

    metric = BinaryAUROC()
    test_preds = torch.Tensor(test_preds)
    correct_preds = torch.Tensor(correct_preds)
    metric.update(test_preds, correct_preds)
    print(f"BinaryAUROC result: {metric.compute()}\n")


if __name__ == "__main__":
    data: MembershipDataset = torch.load(MEMBERSHIP_DATASET_PATH)
    
    train_data = MembershipDataset()
    test_data = MembershipDataset()

    train_data.ids = data.ids[:10000]
    train_data.imgs = data.imgs[:10000]
    train_data.labels = data.labels[:10000]
    train_data.membership = data.membership[:10000]

    test_data.ids = data.ids[10000:]
    test_data.imgs = data.imgs[10000:]
    test_data.labels = data.labels[10000:]
    test_data.membership = data.membership[10000:]


    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
    model = NeuralNetwork().to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1e-5)
    loss_fn = nn.MSELoss()

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model)

    print("Done!")

