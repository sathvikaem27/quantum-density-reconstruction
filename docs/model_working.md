# Model Working

## Overview

The objective of this project is to reconstruct a valid quantum density matrix from measurement data using a neural network model, while strictly enforcing all physical constraints required by quantum mechanics.

A density matrix ρ must satisfy the following properties:
- Hermiticity: ρ = ρ†
- Positive Semi-Definiteness: ρ ≥ 0
- Unit Trace: Tr(ρ) = 1

Instead of enforcing these constraints using penalty terms, the model guarantees physical validity by construction.

## Density Matrix Parametrization

The neural network predicts a lower triangular matrix L.  
The density matrix is reconstructed as:

\[
\rho = \frac{LL^\dagger}{\mathrm{Tr}(LL^\dagger)}
\]

This parametrization ensures:
- **Hermiticity**, since \( LL^\dagger \) is Hermitian
- **Positive Semi-Definiteness**, since \( LL^\dagger \) is PSD by definition
- **Unit Trace**, enforced by explicit normalization

This approach avoids post-hoc correction and guarantees physically valid outputs for all inputs.
## Neural Network Architecture

A feed-forward neural network is used to map measurement data to the parameters of the lower triangular matrix L.

The architecture consists of:
- Fully connected layers
- ReLU non-linearities
- A final linear projection producing the entries of L

The lower triangular structure is enforced explicitly during the forward pass.
## Training Objective

The model is trained to minimize the mean squared error between the reconstructed density matrix and the ground truth density matrix generated from synthetic measurement data.
## Evaluation Metrics

The trained model is evaluated using:
- **Quantum Fidelity**, measuring similarity between reconstructed and target states
- **Trace Distance**, measuring distinguishability between quantum states

These metrics provide complementary insights into reconstruction accuracy.
