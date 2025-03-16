import os
import csv

import torch
import pandas as pd
from tqdm import tqdm


def get_true_label(path):
    if type(path) == tuple:
        path = path[0]
    filename = path.split('/')[-1]
    parts = filename.split('_')
    return int(parts[-1][0])


def get_acc(path_to_csv):
    df = pd.read_csv(path_to_csv)
    correct_predictions = (df['true_label'] == df['predicted_label']).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions if total_predictions != 0 else 0
    
    return accuracy


def process_data(dataloader, model, device, output_folder):
    model.eval()
    results = []
    
    csv_file = os.path.join(output_folder, 'predictions.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_path', 'true_label', 'predicted_label', 'confidence'])

        for img, path in tqdm(dataloader, desc="Processing images"):
            img = img.to(device)
            true_label = get_true_label(path)

            with torch.no_grad():
                output = model(img)
                _, predicted = torch.max(output, 1)  # Get the index of the highest logit
                confidence = torch.nn.functional.softmax(output, dim=1)[0, predicted].item()

            predicted_label = predicted.item()
            results.append({
                'image_path': path,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'confidence': confidence
            })

            writer.writerow([path, true_label, predicted_label, confidence])

    print(f"Results saved to {csv_file}")