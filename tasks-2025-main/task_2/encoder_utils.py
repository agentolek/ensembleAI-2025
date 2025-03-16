import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from typing import Tuple


BATCH_SIZE = 64
output_dim = 1024
mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]


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


torch.serialization.add_safe_globals([TaskDataset])


class SimCLRTransform:
    def __init__(self):
        self.base_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=32),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        img1 = self.base_transform(img)
        img2 = self.base_transform(img)
        return img1, img2


class SimCLR(torch.nn.Module):
    def __init__(self, base_model, feature_dim=1024, projection_dim=128):
        super().__init__()

        self.encoder = base_model

        # Projection Head
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        features = self.encoder(x)  # 1024-features vector
        projections = self.projection_head(features)
        return features, projections


def load_model():
    model = torchvision.models.resnet50(pretrained=True)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(in_features=2048, out_features=output_dim)
    return model


def get_dataloader():
    data = torch.load("ModelStealingPub.pt", weights_only=False)
    data.transform = SimCLRTransform()
    simclt_dataloader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    return simclt_dataloader


def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)

    sim = torch.nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

    labels = torch.arange(batch_size, device=z_i.device)
    labels = torch.cat([labels, labels], dim=0)

    sim /= temperature
    loss = torch.nn.functional.cross_entropy(sim, labels)
    return loss
