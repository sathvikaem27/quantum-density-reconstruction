import torch
from model import DensityMatrixModel

model = DensityMatrixModel(input_dim=10, dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

dummy_x = torch.randn(100, 10)
dummy_y = torch.eye(2).repeat(100, 1, 1)

for epoch in range(50):
    optimizer.zero_grad()
    pred = model(dummy_x)
    loss = torch.mean((pred - dummy_y) ** 2)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "../outputs/model.pt")
