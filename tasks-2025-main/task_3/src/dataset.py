from typing import Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TaskDataset(Dataset):
    def __init__(self):

        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.ToTensor(),
        ])

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)
    

def train_test_split(dataset, test_size=0.2):
    n = len(dataset)
    n_test = int(n * test_size)
    n_train = n - n_test
    train_dataset = TaskDataset()
    test_dataset = TaskDataset()
    train_dataset.ids = dataset.ids[:n_train]
    train_dataset.imgs = dataset.imgs[:n_train]
    train_dataset.labels = dataset.labels[:n_train]
    test_dataset.ids = dataset.ids[n_train:]
    test_dataset.imgs = dataset.imgs[n_train:]
    test_dataset.labels = dataset.labels[n_train:]
    return train_dataset, test_dataset