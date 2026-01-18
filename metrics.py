import torch

def fidelity(rho, sigma):
    sqrt_rho = torch.linalg.matrix_power(rho, 1)
    product = sqrt_rho @ sigma @ sqrt_rho
    return torch.real(torch.trace(torch.linalg.matrix_power(product, 1)))

def trace_distance(rho, sigma):
    diff = rho - sigma
    return 0.5 * torch.linalg.norm(diff, ord='nuc')
