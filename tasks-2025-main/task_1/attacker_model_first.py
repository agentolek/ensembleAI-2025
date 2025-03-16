import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import requests
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torchvision import models
from dotenv import load_dotenv

#  Załaduj zmienne środowiskowe
load_dotenv()
TOKEN = os.environ["ATHENA_TOKEN"]
URL = "149.156.182.9:6060/task-1/submit"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2

# Ścieżki do zbiorów danych
MEMBERSHIP_DATASET_PATH = "C:/Hackathons/ensembleAI-2025/tasks-2025-main/task_1/pub.pt"
PRIVATE_DATASET_PATH = "C:/Hackathons/ensembleAI-2025/tasks-2025-main/task_1/priv_out_.pt"

#  Model ResNet18
def load_resnet18():
    model = models.resnet18(weights=None)  #nie wczytujemy waf
    model.fc = nn.Linear(model.fc.in_features, 44)  # Dopasowanie do liczby klas
    model.to(DEVICE)
    model.eval()  # model w trybie ewaluacji
    return model

# 🔹 Funkcja do wyciągania cech
def extract_features(dataset, model):
    features, labels = [], [] # lables to membership
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for _, img, membership in dataloader:
            img = img.to(DEVICE)
            logits = model(img)

            softmax_probs = F.softmax(logits, dim=1).cpu().numpy()   # prawdopodobieństwa przynależności do klas
            entropy = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=1)  # niepewnosc predykcjo
            loss = F.cross_entropy(logits, torch.tensor(membership).to(DEVICE), reduction='none').cpu().numpy()  # blad klasyfikacji dla danej probki

            # zapisujemy features dla kazdej probki
            for i in range(len(img)):
                features.append([softmax_probs[i].max(), entropy[i], loss[i]])
                labels.append(membership[i])

    return np.array(features), np.array(labels)

# 🔹 Wczytaj dane
public_dataset = torch.load(MEMBERSHIP_DATASET_PATH, weights_only=False)
private_dataset = torch.load(PRIVATE_DATASET_PATH)

# 🔹 Załaduj model ResNet18 i wyciągnij cechy
model = load_resnet18()
X_train, y_train = extract_features(public_dataset, model)

# 🔹 Trenuj model atakujący (SVM)
attack_model = SVC(probability=True)
attack_model.fit(X_train, y_train)

# 🔹 Przewiduj membership score dla prywatnego zbioru
X_private, _ = extract_features(private_dataset, model)
membership_scores = attack_model.predict_proba(X_private)[:, 1]

# 🔹 Zapisz wyniki
df = pd.DataFrame({"ids": private_dataset.ids, "score": membership_scores})
df.to_csv("submission.csv", index=False)

# 🔹 Wyślij wyniki
result = requests.post(
    URL,
    headers={"token": TOKEN},
    files={"csv_file": ("submission.csv", open("submission.csv", "rb"))}
)
print(result.status_code, result.text)
