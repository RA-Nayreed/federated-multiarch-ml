"""
Utility functions for Federated Learning.

This module provides essential utility functions for federated learning experiments
including data loading, model parameter management, learning rate scheduling,
data distribution, and model persistence.

Key Functions:
    - Data loading and preprocessing for MNIST and CIFAR-10
    - IID and Non-IID data distribution among clients
    - Model parameter extraction and setting
    - Learning rate scheduling with warmup and cosine decay
    - Model saving and loading with metadata
    - Metrics aggregation for federated learning
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os, warnings
from collections import OrderedDict
from typing import Dict, List, Tuple, Any
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def get_parameters(net: torch.nn.Module) -> List[np.ndarray]:
    """
    Extract model parameters as a list of NumPy arrays.
    
    Args:
        net: PyTorch neural network model
        
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Update model parameters from a list of NumPy arrays.
    
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def create_lr_scheduler(optimizer: torch.optim.Optimizer, args: Any, 
                       num_training_batches_per_epoch: int) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a learning rate scheduler with warmup and cosine decay.
        
    """
    if not hasattr(args, 'local_ep') or args.local_ep <= 0:
        raise ValueError(f"Invalid local epochs configuration: {getattr(args, 'local_ep', None)}")
    if num_training_batches_per_epoch <= 0:
        raise ValueError(f"Invalid batches per epoch: {num_training_batches_per_epoch}")

    total_steps = args.local_ep * num_training_batches_per_epoch
    if total_steps <= 0:
        raise ValueError(f"Invalid total steps configuration: {total_steps}")

    warmup_epochs = getattr(args, 'warmup_epochs', 0)
    if warmup_epochs < 0:
        raise ValueError(f"Invalid warmup epochs: {warmup_epochs}")
    
    warmup_steps = int(warmup_epochs * num_training_batches_per_epoch)
    
    min_training_steps = max(1, int(0.1 * total_steps))
    warmup_steps = min(warmup_steps, total_steps - min_training_steps)

    if warmup_steps > 0:
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=1e-6, 
            end_factor=1.0, 
            total_iters=warmup_steps
        )
    else:
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=1.0, 
            end_factor=1.0, 
            total_iters=1
        )
        warmup_steps = 1

    cosine_steps = total_steps - warmup_steps
    if cosine_steps < min_training_steps:
        raise ValueError(f"Invalid scheduler configuration: Too few steps for cosine annealing ({cosine_steps})")

    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=getattr(args, 'lr_min', 1e-6)
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps]
    )
        
    return scheduler


def load_data(dataset: str) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Load MNIST or CIFAR-10 dataset with appropriate preprocessing.

    """
    warnings.filterwarnings('ignore', category=DeprecationWarning, 
                          message="'mode' parameter is deprecated and will be removed in Pillow 13")
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


def distribute_data(train_data: torch.utils.data.Dataset, num_clients: int, 
                   iid: bool = True) -> Dict[int, set]:
    """
    Distribute data among clients using IID or Non-IID distribution.
    
    This function partitions the training dataset among multiple clients.
    For IID distribution, data is randomly shuffled and evenly distributed.
    For Non-IID distribution, data is partitioned by class labels to simulate
    real-world data heterogeneity.
    
    """
    if num_clients <= 0:
        raise ValueError(f"Number of clients must be positive, got {num_clients}")
    if len(train_data) < num_clients:
        raise ValueError(f"Not enough data ({len(train_data)}) for {num_clients} clients")
    
    min_samples_per_client = 1
    if len(train_data) // num_clients < min_samples_per_client:
        print(f"Warning: Very few samples per client ({len(train_data) // num_clients}). Consider reducing num_clients.")
    
    if iid:
        return distribute_iid(train_data, num_clients)
    else:
        return distribute_non_iid(train_data, num_clients)


def distribute_iid(train_data: torch.utils.data.Dataset, num_clients: int) -> Dict[int, set]:
    """
    Distribute data in IID manner across all clients.
    
    In IID distribution, data is randomly shuffled and evenly distributed
    among clients, ensuring each client has a representative sample of
    the entire dataset.

    """
    num_items = len(train_data) // num_clients
    client_data = {i: set() for i in range(num_clients)}
    all_idxs = list(range(len(train_data)))
    
    for i in range(num_clients):
        num_to_sample = min(num_items, len(all_idxs))
        if num_to_sample == 0:
            continue
        sampled_idxs = np.random.choice(all_idxs, num_to_sample, replace=False)
        client_data[i].update(sampled_idxs)
        all_idxs = list(set(all_idxs) - set(sampled_idxs))
    
    for i, idx in enumerate(all_idxs):
        client_id = i % num_clients
        client_data[client_id].add(idx)
    
    return client_data


def distribute_non_iid(train_data: torch.utils.data.Dataset, num_clients: int) -> Dict[int, set]:
    """
    Distribute data in non-IID manner using shard-based partitioning.
    
    In Non-IID distribution, data is partitioned by class labels to simulate
    real-world data heterogeneity where different clients have different
    class distributions.

    """
    num_shards = max(2, num_clients * 2)
    if num_shards > len(train_data):
        num_shards = len(train_data)
        print(f"Warning: Not enough data for {num_clients * 2} shards. Using {num_shards} shards.")

    num_imgs_per_shard = len(train_data) // num_shards
    remainder = len(train_data) % num_shards
    idx_shard = [i for i in range(num_shards)]
    client_data = {i: set() for i in range(num_clients)}
    labels = np.array([train_data[i][1] for i in range(len(train_data))])
    sorted_idx = np.argsort(labels)

    shards = []
    start_idx = 0
    for i in range(num_shards):
         end_idx = start_idx + num_imgs_per_shard
         if i < remainder:
             end_idx += 1
         shards.append(sorted_idx[start_idx:end_idx])
         start_idx = end_idx

    shards_per_client = 2
    for i in range(num_clients):
        if not idx_shard:
            break
        shards_to_assign = min(shards_per_client, len(idx_shard))
        rand_set = set(np.random.choice(idx_shard, shards_to_assign, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for shard_idx in rand_set:
            client_data[i] = client_data[i].union(set(shards[shard_idx]))

    assigned_indices = set()
    for indices in client_data.values():
        assigned_indices.update(indices)
    all_indices = set(range(len(train_data)))
    unassigned_indices = list(all_indices - assigned_indices)
    if unassigned_indices:
        print(f"Warning: Distributing {len(unassigned_indices)} unassigned data points randomly.")
        for i, idx in enumerate(unassigned_indices):
            client_id = i % num_clients
            client_data[client_id].add(idx)

    return client_data


def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """
    Compute weighted average of metrics across multiple clients.
    
    This function aggregates metrics from multiple clients by computing
    weighted averages based on the number of examples each client processed.

    """
    if not metrics:
        return {}
    total_examples = sum([num_examples for num_examples, _ in metrics])
    if total_examples == 0:
        return {}
    weighted_metrics = {}
    for key in metrics[0][1].keys():
        if key in ['train_loss', 'test_accuracy', 'train_accuracy']:
            try:
                weighted_sum = sum([metric_dict.get(key, 0) * num_examples for num_examples, metric_dict in metrics])
                weighted_metrics[key] = weighted_sum / total_examples
            except (TypeError, ZeroDivisionError):
                pass
    return weighted_metrics


def save_model(model: torch.nn.Module, model_name: str, dataset: str, strategy: str, 
               args: Any, save_dir: str = "./models") -> str:
    """
    Save the trained model with metadata to disk.
    
    This function saves the model state dictionary along with metadata
    including model configuration, training parameters, and experiment settings.
    
    Args:
        model: Trained PyTorch model to save
        model_name: Name of the model architecture
        dataset: Name of the dataset used for training
        strategy: Federated learning strategy used
        args: Configuration arguments used during training
        save_dir: Directory to save the model in

    """
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{model_name}_{dataset}_{strategy}_clients{args.num_users}_rounds{args.epochs}"
    if not args.iid:
        filename += "_noniid"
    filename += ".pth"
    save_path = os.path.join(save_dir, filename)
    
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


def load_model(model_path: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load a saved model from checkpoint file.
    
    This function loads a previously saved model along with its metadata
    and configuration parameters.

    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    if 'model_name' not in checkpoint:
        raise ValueError(f"Invalid checkpoint file: missing 'model_name' in {model_path}")
    if 'dataset' not in checkpoint:
        raise ValueError(f"Invalid checkpoint file: missing 'dataset' in {model_path}")
    
    model_name = checkpoint['model_name']
    dataset = checkpoint['dataset']
    snn_timesteps = checkpoint.get('snn_timesteps', 25)
    
    if snn_timesteps <= 0:
        raise ValueError(f"Invalid SNN timesteps in checkpoint: {snn_timesteps}")
    
    from models import get_model
    try:
        model = get_model(model_name, dataset, snn_timesteps)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        raise RuntimeError(f"Failed to create or load model {model_name} for dataset {dataset}: {e}")
    
    return model, checkpoint