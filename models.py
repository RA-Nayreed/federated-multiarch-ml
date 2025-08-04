"""Neural Network Models for MNIST and CIFAR-10 (CNN, MLP, SNN)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

class SimpleCNN(nn.Module):
    """Improved CNN model for MNIST with better architecture"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CIFAR10CNN(nn.Module):
    """Improved CNN model for CIFAR-10"""
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class MNISTSNN(nn.Module):
    """Spiking Neural Network for MNIST"""
    def __init__(self, num_classes=10, spike_grad=surrogate.fast_sigmoid(slope=25),
                 beta=0.95, threshold=1.0, num_steps=25):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(28*28, 1000)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, threshold=threshold)
        self.fc2 = nn.Linear(1000, 500)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, threshold=threshold)
        self.fc3 = nn.Linear(500, num_classes)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, threshold=threshold, output=True)

    def forward(self, x):
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
    """Improved Spiking Neural Network for CIFAR-10"""
    def __init__(self, num_classes=10, spike_grad=surrogate.fast_sigmoid(slope=25),
                 beta=0.95, threshold=1.0, num_steps=25):
        super().__init__()
        self.num_steps = num_steps
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, threshold=threshold)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, threshold=threshold)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, threshold=threshold)
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256, bias=False)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, threshold=threshold)
        self.fc2 = nn.Linear(256, num_classes, bias=False)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, threshold=threshold, output=True)

    def forward(self, x):
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
    """Multi-Layer Perceptron for both datasets"""
    def __init__(self, input_size, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

def get_model(model_name: str, dataset: str, snn_timesteps: int = 25):
    """Get the appropriate model for the dataset"""
    if model_name == 'mlp':
        if dataset == 'mnist':
            return MLP(28 * 28, 10)
        elif dataset == 'cifar10':
            return MLP(32 * 32 * 3, 10)
    elif model_name == 'cnn':
        if dataset == 'mnist':
            return SimpleCNN(10)
        elif dataset == 'cifar10':
            return CIFAR10CNN(10)
    elif model_name == 'snn':
        if dataset == 'mnist':
            return MNISTSNN(num_classes=10, num_steps=snn_timesteps)
        elif dataset == 'cifar10':
            return CIFAR10SNN(num_classes=10, num_steps=snn_timesteps)
        else:
            raise ValueError("SNN model supports MNIST and CIFAR-10")
    else:
        raise ValueError(f"Model {model_name} not supported")
