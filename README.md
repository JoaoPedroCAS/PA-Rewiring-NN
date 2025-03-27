# PA Rewiring for Neural Network Initialization

This repository implements **Preferential Attachment (PA) Rewiring**, a novel neural network initialization method based on the paper:

> Scabini, L., De Baets, B., & Bruno, O. M. (2023). [Improving deep neural network random initialization through neuronal rewiring](https://arxiv.org/abs/2303.01507). *Neural Networks*.

## Table of Contents
- [Concept Overview](#concept-overview)
- [Mathematical Foundations](#mathematical-foundations)
- [Implementation Details](#implementation-details)
- [Key Features](#key-features)
- [Usage Guide](#usage-guide)
- [Results Interpretation](#results-interpretation)
- [References](#references)

## Concept Overview

Traditional weight initialization methods (like Kaiming or Xavier) focus solely on weight values. PA Rewiring introduces **network topology** into initialization by:

1. Starting with standard Kaiming initialization
2. Rewiring connections based on neuron strength (sum of input weights)
3. Creating a more balanced network structure before training begins

As shown in Scabini's original work ([GitHub](https://github.com/scabini/PArewiring_weights/tree/main)), this approach leads to:

- Faster convergence
- Better gradient flow
- Improved final accuracy

## Mathematical Foundations

### Neuron Strength Definition
For a weight matrix \( W \in \mathbb{R}^{n \times m} \):

$$
\text{Strength}(s_i) = \sum_{j=1}^{m} |w_{ij}|
$$

### Rewiring Algorithm
**Initialize** weights with Kaiming normal distribution
**Accumulate strengths for each neuron:**

$$
s^{(t)} = s^{(t-1)} + W_{:,t-1}
$$

**Compute probabilities:**

$$
P = \frac{s + |\min(s)| + \epsilon}{\sum(s + |\min(s)| + \epsilon)}
$$
   
**Rewire connections** using preferential attachment:
   - Select targets via weighted random sampling
   - Reassign weights sorted by magnitude

### Symmetric Rewiring
The complete process applies rewiring to both:
- Input dimension (columns of \( W \))
- Output dimension (rows of \( W \), via \( W^T \))

## Implementation Details

### Core Components

1. **Weight Initialization** (`kaiming_init`):
```python
def kaiming_init(weights):
    nn.init.kaiming_normal_(weights, mode='fan_in', nonlinearity='relu')
```

2. **PA Rewiring** (`PA_rewiring`):
```python
def PA_rewiring(weights, seed=None):
    # Convert to numpy for precise operations
    weights_np = weights.detach().cpu().numpy()
    
    for t in range(1, weights_np.shape[1]):
        # Update cumulative strength
        s = s + weights_np[:, t-1]
        
        # Calculate attachment probabilities
        P = (s + np.abs(np.min(s)) + 1e-10)
        P /= np.sum(P)
        
        # Perform rewiring
        targets = rng.choice(n_i, size=n_i, replace=False, p=P)
        new_edges = np.argsort(weights_np[:, t])
        weights_np[targets, t] = weights_np[new_edges, t]
    
    # Update original tensor
    weights.copy_(torch.from_numpy(weights_np))
```

### Network Architecture
The MLP structure follows Scabini's experimental setup:
```python
class MLP(nn.Module):
    def __init__(self, n_neurons=500):
        super().__init__()
        self.layer1 = nn.Linear(784, n_neurons)  # Input layer
        self.layer2 = nn.Linear(n_neurons, n_neurons)  # Hidden layer
        self.output = nn.Linear(n_neurons, 10)  # Output layer
        
        # Initialize with rewiring
        with torch.no_grad():
            kaiming_init(self.layer1.weight)
            symmetric_PA_rewiring(self.layer1.weight)
            ...
```

## Key Features

1. **Topology-Aware Initialization**
   - Goes beyond weight magnitude distribution
   - Optimizes connection patterns before training

2. **Theoretical Guarantees**
   - Preserves Kaiming's variance scaling
   - Gradually reduces strength variance

3. **Computational Efficiency**
   - Single initialization cost
   - No overhead during training

4. **Plug-and-Play**
   - Compatible with any PyTorch model
   - Works alongside existing techniques

## Usage Guide

### Basic Implementation
```python
from model import MLP

# Initialize network with PA rewiring
model = MLP(n_neurons=500)

# Standard training loop
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    # Training code...
```

### Customizing Rewiring
```python
# For custom layers
class CustomLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(100, 100))
        
        # Initialize with rewiring
        kaiming_init(self.weight)
        symmetric_PA_rewiring(self.weight)
```

## References

1. Original Paper:
   ```bibtex
   @article{scabini2023improving,
     title={Improving deep neural network random initialization through neuronal rewiring},
     author={Scabini, Leonardo and De Baets, Bernard and Bruno, Odemir M},
     journal={Neural Networks},
     year={2023}
   }
   ```

2. Implementation Resources:
   - [Original GitHub](https://github.com/scabini/PArewiring_weights)
   - [PyTorch Docs](https://pytorch.org/docs/stable/nn.init.html)
