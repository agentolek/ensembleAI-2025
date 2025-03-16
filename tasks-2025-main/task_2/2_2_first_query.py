import base64
import io
import json
import numpy as np
import onnxruntime as ort
import pickle
import requests
import torch
import torch.nn as nn
import torchvision
from PIL import Image

from torch.utils.data import Dataset
from typing import Tuple


TOKEN = "aHX36NuduY3bHcPVeIJMbuW5X0ZBUS"
SUBMIT_URL = "http://149.156.182.9:6060/task-2/submit"
RESET_URL = "http://149.156.182.9:6060/task-2/reset"
QUERY_URL = "http://149.156.182.9:6060/task-2/query"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path):
    model = torchvision.models.resnet50(weights=None)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(in_features=2048, out_features=1024)
    model.load_state_dict(torch.load(path))
    return model


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


def generate_random_image():
    """Generates a random 32x32 RGB image and returns it as bytes."""
    random_pixels = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(random_pixels, 'RGB')

    # Save to a BytesIO buffer
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    return img_bytes.getvalue()


def get_bytes(single_element):
    subset = single_element[1]  # Modify this if the data is not a tensor
    print(subset)

    img_bytes = io.BytesIO()
    subset.save(img_bytes, format="PNG")
    byte_data = img_bytes.getvalue()
    return byte_data


def load_images():
    dataset = torch.load('/net/tscratch/people/tutorial040/task2/ModelStealingPub.pt', weights_only=False)
    return dataset


def get_images():
    dataset = torch.load('/net/tscratch/people/tutorial040/task2/ModelStealingPub.pt', weights_only=False)
    subset = dataset[0][1]  # Modify this if the data is not a tensor
    print(subset)
    # Create a BytesIO object to hold the byte data
    # Save the sliced dataset to the BytesIO object in .pt format
    img_bytes = io.BytesIO()

    # Save the image to the BytesIO object in PNG format (you can choose other formats like JPEG)
    subset.save(img_bytes, format="PNG")

    # Get the byte data
    byte_data = img_bytes.getvalue()
    # Get the byte data
    return byte_data


def quering_random():
    files = [("files", ("image2.png", generate_random_image(), "image/png")) for _ in range(10)]
    response = requests.post(
        QUERY_URL,
        headers={"token": TOKEN},
        files=files
    )
    if response.status_code == 200:
        buffer = io.BytesIO(response.content)
        np_array = np.load(buffer)
        print(np_array.shape)
        print(np_array)
    else:
        print(response.text)


def reset_example():
    response = requests.post(
        RESET_URL,
        headers={"token": TOKEN}
    )

    print(response.text)


def quering_example(data_element):
    # dataset = load_images()
    # files = [("files", ("image2.png", get_bytes(dataset[i]), "image/png")) for i in range(n)]
    files = ("files", ("image2.png", get_bytes(data_element), "image/png"))
    response = requests.post(
        QUERY_URL,
        headers={"token": TOKEN},
        files=files
    )
    if response.status_code == 200:
        buffer = io.BytesIO(response.content)
        np_array = np.load(buffer)
    #     print(np_array.shape)
    #     print(np_array)
    # else:
    #     print(response.text)
    return np_array


def augmentation(element):
    # Placeholder - symulacja augmentacji (np. dodanie losowego szumu)
    # Zwraca zmodyfikowaną wersję elementu
    return element + torch.randn_like(element) * 0.1  # Przykład: dodanie szumu


def train_with_queries(n):
    # wysyłam query z każdego elementu datasetu
    # dla każdego query robię n augmentacji
    # przepuszczam te augemntacje przez model, mierzę loss i robię backprop
    # zwracam model

    model = load_model("/net/tscratch/people/tutorial040/task2/simclr.pth")

    # dataset = load_images()
    # for element in dataset:
    #     query_result = quering_example(element)
    #     for i in range(n):
    #         aug_item = augmentation(element)
    #         # aug_result = model.train(aug_item)
    #         # loss = torch.norm(query_result - aug_result, p=2), l2_distance
    #         # model.backpropagation
    # # return model

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()  # L2 loss (MSE)

    # Załaduj dane
    dataset = load_images()

    # Trening
    # model.train()  # Ustaw model w tryb treningu

    for element in dataset:
        # Querying example
        query_result = quering_example(element)

        for i in range(n):
            # Augmentacja
            aug_item = augmentation(element)

            # Forward pass
            aug_result = model(aug_item)

            # Obliczanie straty (L2 distance)
            loss = criterion(aug_result, query_result)

            # Backpropagacja
            optimizer.zero_grad()  # Zerowanie gradientów
            loss.backward()  # Obliczanie gradientów
            optimizer.step()  # Aktualizacja wag

            print(f"Strata po augmentacji {i+1}: {loss.item()}")

    return model


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 3, 1024)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)


def submitting_example():
    model = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 3, 1024))

    path = 'dummy_submission.onnx'

    torch.onnx.export(
        model,
        torch.randn(1, 3, 32, 32),
        path,
        export_params=True,
        input_names=["x"],
    )

    with open(path, "rb") as f:
        model = f.read()
        try:
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 1
            stolen_model = ort.InferenceSession(model, sess_options=session_options)
        except Exception as e:
            raise Exception(f"Invalid model, {e=}")
        try:
            out = stolen_model.run(
                None, {"x": np.random.randn(1, 3, 32, 32).astype(np.float32)}
            )[0][0]
        except Exception as e:
            raise Exception(f"Some issue with the input, {e=}")
        assert out.shape == (1024,), "Invalid output shape"

    response = requests.post(SUBMIT_URL, headers={"token": TOKEN}, files={"onnx_model": open(path, "rb")})
    print(response.status_code, response.text)


def submitting_model():
    # model = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 3, 1024))
    model = train_with_queries(5)
    path = 'for_submission_model.onnx'

    torch.onnx.export(
        model,
        torch.randn(1, 3, 32, 32),
        path,
        export_params=True,
        input_names=["x"],
    )

    with open(path, "rb") as f:
        model = f.read()
        try:
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 1
            stolen_model = ort.InferenceSession(model, sess_options=session_options)
        except Exception as e:
            raise Exception(f"Invalid model, {e=}")
        try:
            out = stolen_model.run(
                None, {"x": np.random.randn(1, 3, 32, 32).astype(np.float32)}
            )[0][0]
        except Exception as e:
            raise Exception(f"Some issue with the input, {e=}")
        assert out.shape == (1024,), "Invalid output shape"

    # response = requests.post(SUBMIT_URL, headers={"token": TOKEN}, files={"onnx_model": open(path, "rb")})
    # print(response.status_code, response.text)
    print("Out")


if __name__ == '__main__':
    # reset_example()
    # quering_example()
    # quering_random()
    # submitting_example()
    submitting_model()
