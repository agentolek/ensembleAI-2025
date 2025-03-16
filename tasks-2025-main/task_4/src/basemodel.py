import torch
from torch import nn
import torchvision.models as models


def get_basemodel():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 4)
    
    model.load_state_dict(torch.load('models/resnet50_backdoored.pth'))
    return model


def save_to_onnx(model, path)
    model.eval()
    dummy_input = torch.randn(1, 3, 64, 64)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX model saved at {onnx_model_path}")


def save_weights(model, path)
    torch.save(model.state_dict(), path)



