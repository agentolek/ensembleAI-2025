import torch
from encoder_utils import load_model, get_dataloader, SimCLR, nt_xent_loss, TaskDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


encoder = load_model().to(device)
simclr = SimCLR(encoder).to(device)
optimizer = torch.optim.Adam(simclr.parameters(), lr=3e-4, weight_decay=1e-4)
simclt_dataloader = get_dataloader()


EPOCHS = 10
for epoch in range(EPOCHS):
    print("===")
    for idx, (img1, img2), label in simclt_dataloader:
        img1 = img1.to(device)
        img2 = img2.to(device)

        _, z_i = simclr(img1)
        _, z_j = simclr(img2)

        loss = nt_xent_loss(z_i, z_j).to(device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


torch.save(simclr.state_dict(), 'simclr.pth')
