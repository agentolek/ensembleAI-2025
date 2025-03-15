import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, fname))]
        self.true_labels =[get_true_label(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        true_label = self.true_labels[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, img_path #, true_label