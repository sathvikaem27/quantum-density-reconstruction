import torch
import torch.nn as nn

class DensityMatrixModel(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.dim = dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, dim * dim)
        )

    def forward(self, x):
        L_flat = self.fc(x)
        L = L_flat.view(-1, self.dim, self.dim)
        L = torch.tril(L)

        rho = L @ L.transpose(-1, -2)

        trace = rho.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
        rho = rho / trace.unsqueeze(-1).unsqueeze(-1)

        return rho


