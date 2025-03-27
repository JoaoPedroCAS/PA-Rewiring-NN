import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import kurtosis
from tqdm import tqdm

# ====================== PA Rewiring Functions ==============================
def kaiming_init(weights):
    """Kaiming normal initialization"""
    nn.init.kaiming_normal_(weights, mode='fan_in', nonlinearity='relu')

def PA_rewiring(weights, seed=None):
    """Exact implementation of the pseudocode"""
    n_i, n_j = weights.shape
    rng = np.random.default_rng(seed)
    weights_np = weights.detach().cpu().numpy()  # Fixed: use detach() before numpy()
    
    s = np.zeros(n_i)  # Temporary strength tracking
    
    for t in range(1, n_j):  # t from 1 to n_j-1
        s = s + weights_np[:, t-1]  # Update cumulative strength
        P = s + np.abs(np.min(s)) + 1e-10  # Calculate probabilities
        P = P / np.sum(P)
        
        targets = rng.choice(n_i, size=n_i, replace=False, p=P)
        new_edges = np.argsort(weights_np[:, t])
        weights_np[targets, t] = weights_np[new_edges, t]
    
    with torch.no_grad():
        weights.copy_(torch.from_numpy(weights_np))
    return weights

def symmetric_PA_rewiring(weights, seed=None):
    """Apply rewiring to both input and output dimensions"""
    PA_rewiring(weights, seed)
    PA_rewiring(weights.T, seed)
    return weights

# ====================== MLP Model ==============================
class MLP(nn.Module):
    def __init__(self, n_neurons=500, seed=None):
        super().__init__()
        self.layer1 = nn.Linear(784, n_neurons)
        self.layer2 = nn.Linear(n_neurons, n_neurons)
        self.output = nn.Linear(n_neurons, 10)
        
        # Initialize with rewiring
        with torch.no_grad():
            kaiming_init(self.layer1.weight)
            kaiming_init(self.layer2.weight)
            kaiming_init(self.output.weight)
            
            # Apply rewiring only at initialization
            symmetric_PA_rewiring(self.layer1.weight, seed)
            symmetric_PA_rewiring(self.layer2.weight, seed)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output(x)
    
    def calculate_strengths(self):
        """Calculate neuron strengths (sum of connected weights)"""
        strengths = {}
        with torch.no_grad():
            # Input strengths (sum columns)
            strengths['layer1_input'] = torch.sum(self.layer1.weight, dim=0).cpu().numpy()
            strengths['layer2_input'] = torch.sum(self.layer2.weight, dim=0).cpu().numpy()
            # Output strengths (sum rows)
            strengths['layer1_output'] = torch.sum(self.layer1.weight, dim=1).cpu().numpy()
            strengths['layer2_output'] = torch.sum(self.layer2.weight, dim=1).cpu().numpy()
        return strengths

# ====================== Plotting Functions ==============================
def plot_weight_distributions(original_weights, rewired_weights, layer_name):
    """Fig 1a: Weight value distributions"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.hist(original_weights.flatten(), bins=100, alpha=0.7, color='blue', density=True)
    plt.title(f'Original Weights - {layer_name}')
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    
    plt.subplot(122)
    plt.hist(rewired_weights.flatten(), bins=100, alpha=0.7, color='red', density=True)
    plt.title(f'Rewired Weights - {layer_name}')
    plt.xlabel('Weight Value')
    
    plt.tight_layout()
    plt.savefig(f'plots/weight_distributions_{layer_name}.png')
    plt.close()

def plot_strength_distributions(original_strengths, rewired_strengths, layer_name):
    """Fig 1b: Strength distributions"""
    plt.figure(figsize=(8, 5))
    
    plt.hist(original_strengths, bins=50, alpha=0.5, color='blue', label='Original', density=True)
    plt.hist(rewired_strengths, bins=50, alpha=0.5, color='red', label='Rewired', density=True)
    plt.title(f'Neuron Strength Distribution - {layer_name}')
    plt.xlabel('Strength (sum of weights)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'plots/strength_distributions_{layer_name}.png')
    plt.close()

def plot_max_strengths(original_max, rewired_max):
    """Fig 1c: Maximum absolute strengths"""
    plt.figure(figsize=(8, 5))
    
    plt.boxplot([original_max, rewired_max], labels=['Original', 'Rewired'])
    plt.title('Maximum Absolute Strengths Comparison')
    plt.ylabel('Maximum Strength')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('plots/max_strengths_comparison.png')
    plt.close()

def plot_network_properties(variances, kurtoses, accuracies):
    """Fig 3: Statistical properties vs accuracy"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.scatter(variances, accuracies, alpha=0.7)
    plt.title('Strength Variance vs Accuracy')
    plt.xlabel('Variance of Neuron Strengths')
    plt.ylabel('Test Accuracy')
    
    plt.subplot(122)
    plt.scatter(kurtoses, accuracies, alpha=0.7)
    plt.title('Strength Kurtosis vs Accuracy')
    plt.xlabel('Kurtosis of Neuron Strengths')
    
    plt.tight_layout()
    plt.savefig('plots/network_properties_vs_accuracy.png')
    plt.close()

# ====================== Training and Analysis ==============================
def analyze_network(model):
    """Calculate network statistics"""
    strengths = model.calculate_strengths()
    all_strengths = np.concatenate(list(strengths.values()))
    return {
        'variance': np.var(all_strengths),
        'kurtosis': kurtosis(all_strengths),
        'max_strength': np.max(np.abs(all_strengths))
    }

def train_and_analyze(model, train_loader, test_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Evaluation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        train_losses.append(running_loss/len(train_loader))
        test_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}/{epochs} | Loss: {train_losses[-1]:.4f} | Acc: {accuracy:.4f}')
    
    return train_losses, test_accuracies

def main():
    # Create directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Load FashionMNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.FashionMNIST('./data_fashion', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST('./data_fashion', train=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    # ========== Fig 1: Weight and Strength Distributions ==========
    print("Generating weight and strength distribution plots...")
    # Create original and rewired versions of layer1
    original_layer = nn.Linear(784, 500)
    kaiming_init(original_layer.weight)
    original_weights = original_layer.weight.detach().clone().numpy()
    
    rewired_layer = nn.Linear(784, 500)
    rewired_layer.weight.data = torch.from_numpy(original_weights.copy())
    symmetric_PA_rewiring(rewired_layer.weight)
    rewired_weights = rewired_layer.weight.detach().numpy()
    
    # Plot weight distributions
    plot_weight_distributions(original_weights, rewired_weights, 'Layer1')
    
    # Plot strength distributions
    original_strengths = np.sum(original_weights, axis=0)
    rewired_strengths = np.sum(rewired_weights, axis=0)
    plot_strength_distributions(original_strengths, rewired_strengths, 'Layer1_Input')
    
    # ========== Fig 1c: Max Strengths Comparison ==========
    print("Analyzing maximum strengths...")
    n_trials = 100
    original_max = []
    rewired_max = []
    
    for _ in tqdm(range(n_trials), desc="Max Strength Analysis"):
        # Original
        layer = nn.Linear(100, 100)
        kaiming_init(layer.weight)
        strengths = torch.sum(layer.weight, dim=0).abs().detach().numpy()
        original_max.append(np.max(strengths))
        
        # Rewired
        symmetric_PA_rewiring(layer.weight)
        strengths = torch.sum(layer.weight, dim=0).abs().detach().numpy()
        rewired_max.append(np.max(strengths))
    
    plot_max_strengths(original_max, rewired_max)
    
    # ========== Fig 3: Network Properties vs Accuracy ==========
    print("Analyzing network properties vs accuracy...")
    n_models = 20  # Reduced for faster execution
    variances = []
    kurtoses = []
    accuracies = []
    
    for _ in tqdm(range(n_models), desc="Property vs Accuracy Analysis"):
        model = MLP()
        stats = analyze_network(model)
        variances.append(stats['variance'])
        kurtoses.append(stats['kurtosis'])
        
        # Quick training for demonstration
        _, test_acc = train_and_analyze(model, train_loader, test_loader, epochs=5)
        accuracies.append(test_acc[-1])  # Final accuracy
    
    plot_network_properties(variances, kurtoses, accuracies)
    
    # ========== Training Curve Example ==========
    print("Training final model...")
    model = MLP()
    train_losses, test_accuracies = train_and_analyze(model, train_loader, test_loader, epochs=20)
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Training Curve')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/training_curve.png')
    plt.close()

if __name__ == '__main__':
    main()