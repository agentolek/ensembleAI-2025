import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from typing import Tuple
import torchvision.transforms as transforms


# Ustawienia
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
MEMBERSHIP_DATASET_PATH = "C:/Hackathons/ensembleAI-2025/tasks-2025-main/task_1/pub.pt"       # Path to priv_out_.pt
MIA_CKPT_PATH = "C:/Hackathons/ensembleAI-2025/tasks-2025-main/task_1/01_MIA_69.pt"                 # Path to 01_MIA_69.pt

# Definicja transformacji
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Augmentacja
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889])
])

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
        #print("Shape before fc:", x.shape)
        return self.fc(x)


class TaskDataset(torch.utils.data.Dataset):
    def __init__(self, transform=transform):

        self.ids = []
        self.imgs = []
        self.labels = []

        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=transform):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]

torch.serialization.add_safe_globals([MembershipDataset])

# Przygotowanie danych do ataku
class AttackDataset(torch.utils.data.Dataset):
    def __init__(self, victim_model, dataset):
        self.victim_model = victim_model
        self.dataset = dataset
        self.features = []
        self.membership = []
        self.prepare_data()


    def prepare_data(self):
        dataloader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=False)
        tensor_list = []
        for _, img, label, is_member in dataloader:
            img = img.to(DEVICE)
            with torch.no_grad():
                feature = self.victim_model(img)
            temp = torch.cat((torch.flatten(img), torch.flatten(feature), label.to(DEVICE).to(torch.float32)))
            tensor_list.append(temp)
            #self.features.append(torch.flatten(feature))
            #self.features.append(label.to(DEVICE).to(torch.float32))
            self.membership.append(is_member)

        self.features = torch.stack(tensor_list, dim=0)
        #self.features = torch.cat(self.features)
        self.membership = torch.cat(self.membership)

    def __getitem__(self, index):
        return self.features[index], self.membership[index]

    def __len__(self):
        return len(self.membership)


# Trening atakującego modelu
def train_attack_model(train_loader, model, loss_fn, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for features, membership in train_loader:
            features, membership = features.to(DEVICE), membership.to(DEVICE)
            pred = model(features)
            loss = loss_fn(pred, membership.view(-1, 1).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")


if __name__ == "__main__":
    dataset = torch.load(MEMBERSHIP_DATASET_PATH)

    victim_model = models.resnet18().to(DEVICE)
    victim_model.fc = torch.nn.Linear(512, 44).to(DEVICE)
    victim_model.load_state_dict(torch.load(MIA_CKPT_PATH, map_location=DEVICE))
    victim_model.eval()

    attack_dataset = AttackDataset(victim_model, dataset)
    train_loader = DataLoader(attack_dataset, batch_size=BATCH_SIZE, shuffle=True)

    attack_model = AttackModel().to(DEVICE)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(attack_model.parameters(), lr=0.001)

    train_attack_model(train_loader, attack_model, loss_fn, optimizer)
