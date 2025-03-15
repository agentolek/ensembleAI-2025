import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models
from typing import Tuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
MEMBERSHIP_DATASET_PATH = "/net/tscratch/people/tutorial040/task1/pub.pt"
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
            nn.Linear(512, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    

def train_loop(dataloader, model, loss_fn, optimizer):
    # size = len(dataloader.dataset)
    model.train()

    for _, img, lbl, member in dataloader:
        
        img, lbl, member = img.to(DEVICE).to(torch.float32), lbl.to(DEVICE).to(torch.float32), member.to(DEVICE).to(torch.float32)

        X = torch.cat((img.flatten(), lbl))
        
        pred = model(X)
        loss = loss_fn(pred, member)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * BATCH_SIZE + len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for _, img, lbl, member in dataloader:
        
            img, lbl, member = img.to(DEVICE), lbl.to(DEVICE), member.to(DEVICE)

            X = torch.cat((img.flatten(), lbl))
            pred = model(X)
            test_loss += loss_fn(pred, member).item()
            correct += (pred.argmax(1) == member).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


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
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

    print("Done!")

