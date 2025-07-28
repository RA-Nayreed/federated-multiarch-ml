#!/usr/bin/env python3
"""
Fixed Flower-based Federated Learning Implementation for MNIST and CIFAR-10
Implements FedAvg using the Flower framework with support for SNN, CNN, and MLP models
"""
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
import copy
import random
from collections import defaultdict, OrderedDict
import os
import snntorch as snn
from snntorch import surrogate
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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

def load_data(dataset: str):
    """Load MNIST or CIFAR-10 dataset"""
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_data = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
    elif dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_data = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_data = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
    else:
        raise ValueError(f"Dataset {dataset} not supported")
        
    return train_data, test_data

def get_parameters(net) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    """Update model parameters from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def distribute_data(train_data, num_clients: int, iid: bool = True):
    """Distribute data among clients (IID or non-IID)"""
    if iid:
        return distribute_iid(train_data, num_clients)
    else:
        return distribute_non_iid(train_data, num_clients)

def distribute_iid(train_data, num_clients: int):
    """Distribute data in IID manner"""
    num_items = len(train_data) // num_clients
    client_data = {}
    all_idxs = list(range(len(train_data)))
    
    for i in range(num_clients):
        sampled_idxs = np.random.choice(all_idxs, num_items, replace=False)
        client_data[i] = set(sampled_idxs)
        all_idxs = list(set(all_idxs) - client_data[i])
        
    return client_data

def distribute_non_iid(train_data, num_clients: int):
    """Distribute data in non-IID manner"""
    num_shards = num_clients * 2
    num_imgs = len(train_data) // num_shards
    idx_shard = [i for i in range(num_shards)]
    client_data = {i: set() for i in range(num_clients)}
    
    labels = np.array([train_data[i][1] for i in range(len(train_data))])
    
    idxs_labels = np.vstack((range(len(train_data)), labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        
        for rand in rand_set:
            client_data[i] = client_data[i].union(
                set(idxs[rand * num_imgs:(rand + 1) * num_imgs])
            )
            
    return client_data

class FlowerClient(fl.client.NumPyClient):
    """Flower client implementing federated learning."""
    
    def __init__(self, client_id: int, train_data, test_data, client_indices, 
                 model_name: str, dataset: str, args):
        self.client_id = client_id
        self.train_data = train_data
        self.test_data = test_data
        self.client_indices = client_indices
        self.model_name = model_name
        self.dataset = dataset
        self.args = args
        
        self.net = get_model(model_name, dataset, args.snn_timesteps)
        self.net.to(device)
        
        client_dataset = Subset(train_data, list(client_indices))
        self.trainloader = DataLoader(client_dataset, batch_size=args.local_bs, shuffle=True)
        self.testloader = DataLoader(test_data, batch_size=128, shuffle=False)
        
        print(f"Client {client_id}: {len(client_indices)} samples")
    
    def get_parameters(self, config):
        """Return current local model parameters."""
        return get_parameters(self.net)
    
    def fit(self, parameters, config):
        """Train the model on the locally held training set."""
        set_parameters(self.net, parameters)
        
        train_loss, train_accuracy = self.train()
        
        return (
            get_parameters(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss, "train_accuracy": train_accuracy}
        )
    
    def evaluate(self, parameters, config):
        """Evaluate the model on the locally held test set."""
        set_parameters(self.net, parameters)
        
        test_loss, test_accuracy = self.test()
        
        return test_loss, len(self.testloader.dataset), {"test_accuracy": test_accuracy}
    
    def train(self):
        """Train the model locally."""
        self.net.train()
        
        # Adjust optimizer and learning rate based on model type
        if self.model_name == 'snn':
            optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=1e-4)
        else:
            optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        
        criterion = nn.CrossEntropyLoss()
        
        epoch_losses = []
        correct = 0
        total = 0
        
        for epoch in range(self.args.local_ep):
            batch_losses = []
            
            for batch_idx, (data, targets) in enumerate(self.trainloader):
                data, targets = data.to(device), targets.to(device)
                
                optimizer.zero_grad()
                
                if self.model_name == 'snn':
                    spk_rec, mem_rec = self.net(data)
                    outputs = mem_rec[-1]  # Use membrane potential of last time step
                else:
                    outputs = self.net(data)
                
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping for SNN models
                if self.model_name == 'snn':
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                batch_losses.append(loss.item())
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            
            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def test(self):
        """Test the model locally."""
        self.net.eval()
        criterion = nn.CrossEntropyLoss()
        
        correct = 0
        total = 0
        test_loss = 0
        
        with torch.no_grad():
            for data, targets in self.testloader:
                data, targets = data.to(device), targets.to(device)
                
                if self.model_name == 'snn':
                    spk_rec, mem_rec = self.net(data)
                    outputs = mem_rec[-1]  # Use membrane potential of last time step
                else:
                    outputs = self.net(data)
                
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = test_loss / len(self.testloader)
        
        return avg_loss, accuracy

def client_fn(cid: str, train_data, test_data, client_data_dict, model_name: str, dataset: str, args):
    """Create a Flower client."""
    client_id = int(cid)
    client_indices = client_data_dict[client_id]
    client = FlowerClient(client_id, train_data, test_data, client_indices, model_name, dataset, args)
    return client

def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Compute weighted average of metrics."""
    if not metrics:
        return {}
    
    total_examples = sum([num_examples for num_examples, _ in metrics])
    if total_examples == 0:
        return {}
    
    weighted_metrics = {}
    for key in metrics[0][1].keys():
        weighted_sum = sum([metric_dict[key] * num_examples for num_examples, metric_dict in metrics])
        weighted_metrics[key] = weighted_sum / total_examples
    
    return weighted_metrics

def run_simulation(args):
    """Run Flower simulation."""
    print(f"Starting Flower Federated Learning Simulation")
    print(f"Dataset: {args.dataset.upper()}, Model: {args.model.upper()}")
    print(f"Distribution: {'IID' if args.iid else 'Non-IID'}")
    print(f"Clients: {args.num_users}, Rounds: {args.epochs}")
    if args.model == 'snn':
        print(f"SNN Time Steps: {args.snn_timesteps}")
    print("-" * 60)
    
    # Load data and distribute to clients
    train_data, test_data = load_data(args.dataset)
    client_data_dict = distribute_data(train_data, args.num_users, args.iid)
    
    # Initialize model and parameters
    initial_model = get_model(args.model, args.dataset, args.snn_timesteps)
    initial_parameters = get_parameters(initial_model)
    
    # Define strategy
    def fit_config(server_round: int):
        """Return training configuration dict for each round."""
        config = {
            "server_round": server_round,
            "local_epochs": args.local_ep,
        }
        return config
    
    def evaluate_config(server_round: int):
        """Return evaluation configuration dict for each round."""
        return {"server_round": server_round}
    
    class SaveModelStrategy(fl.server.strategy.FedAvg):
        """Custom strategy that saves final parameters"""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.final_parameters = None
        
        def aggregate_fit(self, server_round, results, failures):
            """Aggregate fit results and save final parameters"""
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(
                server_round, results, failures
            )
            # Save the final parameters
            self.final_parameters = aggregated_parameters
            return aggregated_parameters, aggregated_metrics

    strategy = SaveModelStrategy(
        fraction_fit=args.frac,
        fraction_evaluate=1.0,
        min_fit_clients=max(1, int(args.frac * args.num_users)),
        min_evaluate_clients=args.num_users,
        min_available_clients=args.num_users,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # Define client function
    def create_client_fn(context):
        cid = context.node_config["partition-id"]
        client = client_fn(str(cid), train_data, test_data, client_data_dict, args.model, args.dataset, args)
        return client.to_client()
    
    # Determine client resources
    client_resources = {"num_cpus": 1}
    if args.gpu and torch.cuda.is_available():
        client_resources["num_gpus"] = 0.1
    
    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=create_client_fn,
        num_clients=args.num_users,
        config=fl.server.ServerConfig(num_rounds=args.epochs),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={"ignore_reinit_error": True},
    )

    # Get final model parameters from the strategy
    final_model = get_model(args.model, args.dataset, args.snn_timesteps)
    if strategy.final_parameters is not None:
        final_parameters = fl.common.parameters_to_ndarrays(strategy.final_parameters)
        set_parameters(final_model, final_parameters)
        print("Successfully retrieved final model parameters from federated training.")
    else:
        print("Warning: Could not retrieve final model parameters. Saving initial model.")

    # Save the model
    save_model(final_model, args.model, args.dataset, args)
    
    return history

def plot_results(history, args):
    """Plot training results."""
    if not history.metrics_centralized:
        print("No centralized metrics to plot.")
        return
    
    rounds = list(range(1, len(history.metrics_centralized) + 1))
    
    # Extract metrics if available
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for round_metrics in history.metrics_centralized.values():
        if 'train_loss' in round_metrics:
            train_losses.append(round_metrics['train_loss'])
        if 'train_accuracy' in round_metrics:
            train_accuracies.append(round_metrics['train_accuracy'])
        if 'test_accuracy' in round_metrics:
            test_accuracies.append(round_metrics['test_accuracy'])
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    if train_losses:
        axes[0].plot(rounds[:len(train_losses)], train_losses, 'b-', label='Training Loss')
        axes[0].set_title(f'Training Loss - {args.model.upper()} on {args.dataset.upper()}')
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        axes[0].legend()
    
    # Plot accuracies
    if train_accuracies:
        axes[1].plot(rounds[:len(train_accuracies)], train_accuracies, 'g-', label='Training Accuracy')
    if test_accuracies:
        axes[1].plot(rounds[:len(test_accuracies)], test_accuracies, 'r-', label='Test Accuracy')
    
    axes[1].set_title(f'Accuracy - {args.model.upper()} on {args.dataset.upper()}')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig

def save_model(model, model_name, dataset, args, save_dir="./models"):
    """Save the trained model"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create filename with configuration info
    filename = f"{model_name}_{dataset}_clients{args.num_users}_rounds{args.epochs}"
    if not args.iid:
        filename += "_noniid"
    filename += ".pth"
    
    save_path = os.path.join(save_dir, filename)
    
    # Save both model state dict and configuration
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'dataset': dataset,
        'num_classes': 10,
        'snn_timesteps': getattr(args, 'snn_timesteps', 25),
        'args': vars(args)
    }, save_path)
    
    print(f"Model saved to: {save_path}")
    return save_path

def load_model(model_path, device=None):
    """Load a saved model"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate model
    model = get_model(
        checkpoint['model_name'], 
        checkpoint['dataset'], 
        checkpoint.get('snn_timesteps', 25)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    print(f"Model type: {checkpoint['model_name']}, Dataset: {checkpoint['dataset']}")
    
    return model, checkpoint

def main():
    parser = argparse.ArgumentParser(description='Flower Federated Learning with SNN/CNN/MLP')
    
    # Federated learning parameters
    parser.add_argument('--epochs', type=int, default=10, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1.0, help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    
    # Model and dataset parameters
    parser.add_argument('--model', type=str, default='snn', choices=['mlp', 'cnn', 'snn'], help="model")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help="dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--gpu', action='store_true', help='use gpu')
    
    # SNN specific parameters
    parser.add_argument('--snn_timesteps', type=int, default=25, help='number of time steps for SNN')
    
    args = parser.parse_args()
    
    # Adjust learning rates based on model type
    if args.model == 'snn':
        args.lr = 0.001 if args.lr == 0.01 else args.lr  # Only change if default
    elif args.model == 'cnn':
        args.lr = 0.001 if args.lr == 0.01 else args.lr  # Only change if default
    
    print(f"Configuration: {vars(args)}")
    
    # Run simulation
    history = run_simulation(args)
    
    print("\nSimulation completed successfully!")
    
    # Print final results if available
    if history.metrics_centralized:
        final_metrics = list(history.metrics_centralized.values())[-1]
        if 'test_accuracy' in final_metrics:
            print(f"Final Test Accuracy: {final_metrics['test_accuracy']:.2f}%")
        if 'train_accuracy' in final_metrics:
            print(f"Final Training Accuracy: {final_metrics['train_accuracy']:.2f}%")

if __name__ == '__main__':
    main()  