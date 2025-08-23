"""
Neural Network Models for MNIST and CIFAR-10 Datasets.

This module provides implementations of various neural network architectures
including Convolutional Neural Networks (CNN), Multi-Layer Perceptrons (MLP),
and Spiking Neural Networks (SNN) for image classification tasks.

Supported Models:
    - SimpleCNN: CNN architecture optimized for MNIST dataset
    - CIFAR10CNN: CNN architecture optimized for CIFAR-10 dataset
    - MNISTSNN: Spiking Neural Network for MNIST dataset
    - CIFAR10SNN: Spiking Neural Network for CIFAR-10 dataset
    - MLP: Multi-Layer Perceptron for both datasets

Supported Datasets:
    - MNIST: 28x28 grayscale images (784 input features)
    - CIFAR-10: 32x32 color images (3072 input features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from typing import Tuple


class SimpleCNN(nn.Module):
    """
    Convolutional Neural Network optimized for MNIST dataset.
    
        Architecture:
        - 3 convolutional blocks (2 conv layers per block)
        - Batch Normalization + LeakyReLU after each conv
        - Downsampling using stride=2 in the second conv of blocks 2 and 3
        - Dropout2d after each block for regularization
        - Global Average Pooling before fully connected layers
        - Hidden fully connected layer with dropout
        - Output layer with 'num_classes' units

    """
    
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)
        self.bn6 = nn.BatchNorm2d(128)


        self.dropout2d = nn.Dropout2d(0.25)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Hidden FC Layer
        self.fc1 = nn.Linear(128, 128)
        self.dropout_fc = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

        self.act = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        """
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.dropout2d(x)   
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))   
        x = self.dropout2d(x)
        x = self.act(self.bn5(self.conv5(x)))
        x = self.act(self.bn6(self.conv6(x)))
        x = self.dropout2d(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x


class CIFAR10CNN(nn.Module):
    """
    Convolutional Neural Network optimized for CIFAR-10 dataset.
    
    Architecture:
        - Block 1:
            - Conv2d: 3 input channels → 64 channels, kernel size 3, padding 1
            - BatchNorm2d
            - LeakyReLU (negative_slope=0.01)
            - Conv2d: 64 → 128 channels, kernel size 3, stride 2, padding 1
            - BatchNorm2d
            - LeakyReLU
            - Dropout2d (p=0.2)
        - Block 2:
            - Conv2d: 128 → 256 channels, kernel size 3, padding 1
            - BatchNorm2d
            - LeakyReLU
            - Conv2d: 256 → 256 channels, kernel size 3, stride 2, padding 1
            - BatchNorm2d
            - LeakyReLU
            - Dropout2d (p=0.2)
        - Block 3:
            - Conv2d: 256 → 512 channels, kernel size 3, padding 1
            - BatchNorm2d
            - LeakyReLU
            - Conv2d: 512 → 256 channels, kernel size 1
            - BatchNorm2d
            - LeakyReLU
            - Dropout2d (p=0.3)
        - Adaptive Average Pooling to (1, 1)
        - Fully connected layers:
            - Linear: 256 → 192
            - LeakyReLU
            - Dropout (p=0.5)
            - Linear: 192 → 128
            - LeakyReLU
            - Dropout (p=0.5)
            - Linear: 128 → num_classes
    
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout2d(0.2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout2d(0.2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout2d(0.3)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(256, 192)
        self.fc2 = nn.Linear(192, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = self.dropout_fc(x)
        x = F.leaky_relu(self.fc2(x), 0.01)
        x = self.dropout_fc(x)
        x = self.fc3(x)
        return x


class MNISTSNN(nn.Module):
    """
    Spiking Neural Network for MNIST dataset.
    
    Architecture:
        - 3 fully connected layers with Leaky Integrate-and-Fire neurons
        - Temporal dynamics over multiple time steps
        - Surrogate gradient for backpropagation
    
    Args:
        num_classes (int): Number of output classes. Defaults to 10.
        spike_grad: Surrogate gradient function for backpropagation.
                   Defaults to fast_sigmoid with slope=25.
        beta (float): Decay rate for membrane potential. Defaults to 0.95.
        threshold (float): Firing threshold for neurons. Defaults to 1.0.
        num_steps (int): Number of time steps for temporal dynamics.
                        Defaults to 25.
    
    Input Shape:
        (batch_size, 1, 28, 28) - MNIST grayscale images
    
    Output Shape:
        Tuple of (spikes, membrane_potentials) where:
        - spikes: (num_steps, batch_size, num_classes)
        - membrane_potentials: (num_steps, batch_size, num_classes)
    """
    
    def __init__(self, 
                 num_classes: int = 10, 
                 spike_grad = surrogate.fast_sigmoid(slope=25),
                 beta: float = 0.95, 
                 threshold: float = 1.0, 
                 num_steps: int = 25):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(28*28, 1000)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, 
                             init_hidden=False, threshold=threshold)
        self.fc2 = nn.Linear(1000, 500)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, 
                             init_hidden=False, threshold=threshold)
        self.fc3 = nn.Linear(500, num_classes)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, 
                             init_hidden=False, threshold=threshold, output=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the spiking network.

        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk3_rec = []
        mem3_rec = []
        
        for step in range(self.num_steps):
            cur1 = self.fc1(x.view(x.size(0), -1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)
            
        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)


class CIFAR10SNN(nn.Module):
    """
    Spiking Neural Network for CIFAR-10 dataset.
    
    Architecture:
        - 3 convolutional layers with Leaky Integrate-and-Fire neurons
        - 2 fully connected layers with LIF neurons
        - Average pooling between conv layers
        - Temporal dynamics over multiple time steps
    
    Args:
        num_classes (int): Number of output classes. Defaults to 10.
        spike_grad: Surrogate gradient function for backpropagation.
                   Defaults to fast_sigmoid with slope=25.
        beta (float): Decay rate for membrane potential. Defaults to 0.95.
        threshold (float): Firing threshold for neurons. Defaults to 1.0.
        num_steps (int): Number of time steps for temporal dynamics.
                        Defaults to 25.
    
    Input Shape:
        (batch_size, 3, 32, 32) - CIFAR-10 color images
    
    Output Shape:
        Tuple of (spikes, membrane_potentials) where:
        - spikes: (num_steps, batch_size, num_classes)
        - membrane_potentials: (num_steps, batch_size, num_classes)
    """
    
    def __init__(self, 
                 num_classes: int = 10, 
                 spike_grad = surrogate.fast_sigmoid(slope=25),
                 beta: float = 0.95, 
                 threshold: float = 1.0, 
                 num_steps: int = 25):
        super().__init__()
        self.num_steps = num_steps
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, 
                             init_hidden=False, threshold=threshold)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, 
                             init_hidden=False, threshold=threshold)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, 
                             init_hidden=False, threshold=threshold)
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256, bias=False)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, 
                             init_hidden=False, threshold=threshold)
        self.fc2 = nn.Linear(256, num_classes, bias=False)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad, 
                             init_hidden=False, threshold=threshold, output=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the spiking network.

        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        spk5_rec = []
        mem5_rec = []
        
        for step in range(self.num_steps):
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur1_pooled = self.pool(spk1)
            cur2 = self.conv2(cur1_pooled)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur2_pooled = self.pool(spk2)
            cur3 = self.conv3(cur2_pooled)
            spk3, mem3 = self.lif3(cur3, mem3)
            cur3_pooled = self.pool(spk3)
            cur3_flat = cur3_pooled.view(cur3_pooled.size(0), -1)
            cur4 = self.fc1(cur3_flat)
            spk4, mem4 = self.lif4(cur4, mem4)
            cur5 = self.fc2(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)
            spk5_rec.append(spk5)
            mem5_rec.append(mem5)
            
        return torch.stack(spk5_rec, dim=0), torch.stack(mem5_rec, dim=0)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for image classification.
    
    Architecture:
        - 4 fully connected layers with decreasing sizes
        - Dropout for regularization
        - ReLU activation functions

    """
    
    def __init__(self, input_size: int, num_classes: int = 10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        """
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


def get_model(model_name: str, dataset: str, snn_timesteps: int = 25) -> nn.Module:
    """
    Factory function to create appropriate model for given dataset and architecture.
    
    """
    if model_name == 'mlp':
        if dataset == 'mnist':
            return MLP(28 * 28, 10)
        elif dataset == 'cifar10':
            return MLP(32 * 32 * 3, 10)
        else:
            raise ValueError(f"MLP model does not support dataset: {dataset}")
    elif model_name == 'cnn':
        if dataset == 'mnist':
            return SimpleCNN(10)
        elif dataset == 'cifar10':
            return CIFAR10CNN(10)
        else:
            raise ValueError(f"CNN model does not support dataset: {dataset}")
    elif model_name == 'snn':
        if dataset == 'mnist':
            return MNISTSNN(num_classes=10, num_steps=snn_timesteps)
        elif dataset == 'cifar10':
            return CIFAR10SNN(num_classes=10, num_steps=snn_timesteps)
        else:
            raise ValueError(f"SNN model does not support dataset: {dataset}")
    else:
        raise ValueError(f"Model architecture '{model_name}' not supported. "
                        f"Available options: 'mlp', 'cnn', 'snn'")
