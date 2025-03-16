import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader

# Ustawienia
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
MEMBERSHIP_DATASET_PATH = "C:/Hackathons/ensembleAI-2025/tasks-2025-main/task_1/pub.pt"
MIA_CKPT_PATH = "C:/Hackathons/ensembleAI-2025/tasks-2025-main/task_1/01_MIA_69.pt"

# Model ofiary (resnet18)
class VictimModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(512, 44)  # Dopasowanie do liczby klas
        self.hidden_features = None
        self.resnet.layer4.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.hidden_features = output.clone().detach()

    def forward(self, x):
        _ = self.resnet(x)
        return self.hidden_features

# Model atakujący (RMIA)
class AttackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Przygotowanie danych do ataku
class AttackDataset(torch.utils.data.Dataset):
    def __init__(self, victim_model, dataset):
        self.victim_model = victim_model
        self.dataset = dataset
        self.features = []
        self.labels = []
        self.prepare_data()

    def prepare_data(self):
        dataloader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=False)
        for _, img, _, membership in dataloader:
            img = img.to(DEVICE)
            with torch.no_grad():
                feature = self.victim_model(img)
            self.features.append(feature.cpu())
            self.labels.append(membership.cpu())
        self.features = torch.cat(self.features)
        self.labels = torch.cat(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

# Trening atakującego modelu
def train_attack_model(train_loader, model, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            pred = model(features)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

if __name__ == "__main__":
    victim_model = VictimModel().to(DEVICE)
    victim_model.load_state_dict(torch.load(MIA_CKPT_PATH, map_location=DEVICE))
    victim_model.eval()

    dataset = torch.load(MEMBERSHIP_DATASET_PATH)
    attack_dataset = AttackDataset(victim_model, dataset)
    train_loader = DataLoader(attack_dataset, batch_size=BATCH_SIZE, shuffle=True)

    attack_model = AttackModel().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(attack_model.parameters(), lr=0.001)

    train_attack_model(train_loader, attack_model, criterion, optimizer)
    