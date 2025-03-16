import torch
import torchvision
from example_submission import TaskDataset
from SimCLRTransform import SimCLRTransform
from torch.utils.data import DataLoader


# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_dim = 1024

model = torchvision.models.resnet50(pretrained=True)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Linear(in_features=2048, out_features=output_dim)

data = torch.load("ModelStealingPub.pt", weights_only=False)


BATCH_SIZE = 32
data.transform = SimCLRTransform()


simclt_dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True,)


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


encoder = model
simclr = SimCLR(encoder)


def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)

    sim = torch.nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

    labels = torch.arange(batch_size, device=z_i.device)
    labels = torch.cat([labels, labels], dim=0)

    sim /= temperature
    loss = torch.nn.functional.cross_entropy(sim, labels)
    return loss


optimizer = torch.optim.Adam(simclr.parameters(), lr=3e-4, weight_decay=1e-4)

for epoch in range(1):
    for (img1, img2), _ in simclt_dataloader:
        _, z_i = simclr(img1)
        _, z_j = simclr(img2)

        loss = nt_xent_loss(z_i, z_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

