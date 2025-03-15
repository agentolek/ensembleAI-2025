import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import utils
from transform import get_transform
from basemodel import get_basemodel
from dataset import ImageDataset


if __name__ == '__main__':
    train_folder = './train'
    output_folder = './output'

    transform = get_transform()

    dataset = ImageDataset(train_folder, transform=transform)
    dataloader = DataLoader(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_basemodel()
    model.to(device)

    utils.process_data(dataloader, model, device, output_folder)
    print(utils.get_acc('output/predictions.csv'))
