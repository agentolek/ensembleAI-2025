import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optimimpor
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import models

from src.dataset import TaskDataset, train_test_split
from src.PGD_attack import PGDAttack


def save_to_csv(dataset, path):
    ids = []
    labels = []
    for i in range(len(dataset)):
        id_, _, label = dataset[i]
        ids.append(id_)
        labels.append(label)
    df = pd.DataFrame({'id': ids, 'label': labels})
    df.to_csv(path, index=False)


if __name__ == '__main__':
    dataset_path = 'data/train.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = torch.load(dataset_path, weights_only=False)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.8)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for id_, images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
    for id_, images, labels in test_dataloader: 
        images, labels = images.to(device), labels.to(device)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)

    # model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.to(device)

    learning_rate = 1e-5  # Common starting learning rate
    beta1 = 0.9           # Exponential decay rate for first moment estimate
    beta2 = 0.999         # Exponential decay rate for second moment estimate
    epsilon = 1e-8        # Small constant for numerical stability
    weight_decay = 1e-3   # L2 regularization

    # Create the Adam optimizer
    optimizer = optim.Adam(
        model.parameters(),  # Parameters of the model
        lr=learning_rate,    # Learning rate
        betas=(beta1, beta2), # Tuple for beta1 and beta2
        eps=epsilon,         # Epsilon for numerical stability
        weight_decay=weight_decay  # Weight decay (L2 regularization)
    )
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    pgd_attack = PGDAttack(model, eps=0.01, alpha=0.01, iters=15)


    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as tepoch:
            for id_, img, labels in tepoch:
                img, labels = img.to(device), labels.to(device)

                adv_inputs = pgd_attack.perturb(img, labels)

                optimizer.zero_grad()

                outputs_clean = model(img)
                loss_clean = nn.CrossEntropyLoss()(outputs_clean, labels)

                outputs_adv = model(adv_inputs)
                loss_adv = nn.CrossEntropyLoss()(outputs_adv, labels)

                loss = loss_clean + loss_adv

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs_clean.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                tepoch.set_postfix(loss=running_loss / (tepoch.n + 1), accuracy=100. * correct / total)


        model.eval()
        torch.save(model.state_dict(), 'randint50.pt')

        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for id_, inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = 100 * correct_predictions / total_samples
        print(f'Test Accuracy: {accuracy:.2f}%')