
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import weakref
import gc
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple, Any
from models import get_model
from flwr.server.strategy import FedProx, FedAdagrad, FedAdam
from utils import (get_parameters, set_parameters, weighted_average, create_lr_scheduler)

class FlowerClient(fl.client.NumPyClient):
    """
    Flower client implementing federated learning with local training and evaluation.
    
    This client handles:
    - Local model training with various optimizers
    - Model evaluation on local test data
    - Parameter aggregation and communication with server
    - Memory management and cleanup
    
    Args:
        client_id (int): Unique identifier for this client
        train_data: Training dataset
        test_data: Test dataset for evaluation
        client_indices: Indices of training data assigned to this client
        model_name (str): Type of neural network model ('snn', 'cnn', 'mlp')
        dataset (str): Dataset name ('mnist', 'cifar10')
        args: Configuration arguments containing training parameters
    """
    
    def __init__(self, client_id: int, train_data, test_data, client_indices,
                 model_name: str, dataset: str, args):
        self.client_id = client_id
        self.train_data = train_data
        self.test_data = test_data
        self.client_indices = list(client_indices)

        self.model_name = model_name
        self.dataset = dataset
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
        self.net = get_model(model_name, dataset, args.snn_timesteps)
        self.net.to(self.device)
        
        client_dataset = Subset(train_data, self.client_indices)
        if len(client_dataset) == 0:
            print(f"Client {client_id}: WARNING — no training samples for this client.")
            self.trainloader = []  # sentinel for empty train set (train() checks len())
            self.empty_train = True
        else:
            effective_batch_size = min(int(args.local_bs), max(1, len(client_dataset)))
            if effective_batch_size != args.local_bs:
                print(f"Client {client_id}: Reduced batch size from {args.local_bs} to {effective_batch_size} due to small dataset")
            self.trainloader = DataLoader(
                client_dataset,
                batch_size=effective_batch_size,
                shuffle=True,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            self.empty_train = False
        self.testloader = DataLoader(test_data, batch_size=128, shuffle=False,
                                     pin_memory=True if self.device.type == 'cuda' else False)
        print(f"Client {client_id}: {len(client_indices)} samples on device {self.device}")

    def cleanup(self):
        if hasattr(self, 'trainloader'):
            try:
                del self.trainloader
            except Exception:
                self.trainloader = None
        if hasattr(self, 'testloader'):
            try:
                del self.testloader
            except Exception:
                self.testloader = None
        
        if hasattr(self, 'net'):
            try:
                self.net.cpu()
            except Exception:
                pass
            try:
                del self.net
            except Exception:
                self.net = None
            self.net = None
        
        if hasattr(self, 'optimizer'):
            try: del self.optimizer
            except Exception: self.optimizer = None
        if hasattr(self, 'scheduler'):
            try: del self.scheduler
            except Exception: self.scheduler = None
        
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        if not hasattr(self, 'net') or self.net is None:
            raise RuntimeError(f"Client {getattr(self,'client_id', '?')}: get_parameters called but model is not initialized.")
        return get_parameters(self.net)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, float]]:

        set_parameters(self.net, parameters)
        try:
            self.net.to(self.device)
        except Exception:
            pass

        try:
            train_loss, train_accuracy = self.train(parameters)
        except Exception as e:
            print(f"Client {self.client_id}: training failed with error: {e}")
            return get_parameters(self.net), len(self.client_indices), {"train_loss": float('inf'), "train_accuracy": 0.0}

        if hasattr(self, 'trainloader') and hasattr(self.trainloader, 'dataset'):
            num_examples = len(self.trainloader.dataset)
        else:
            num_examples = len(self.client_indices) if hasattr(self, 'client_indices') else 0

        return (
            get_parameters(self.net),
            int(num_examples),
            {"train_loss": float(train_loss), "train_accuracy": float(train_accuracy)}
        )

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, float]]:
        set_parameters(self.net, parameters)
        try:
            self.net.to(self.device)
        except Exception:
            pass
        try:
            test_loss, test_accuracy = self.test()
        except Exception as e:
            print(f"Client {self.client_id}: evaluation failed with error: {e}")
            return float('inf'), 0, {"test_accuracy": 0.0}
        num_test = len(self.testloader.dataset) if hasattr(self.testloader, 'dataset') else 0
        return float(test_loss), int(num_test), {"test_accuracy": float(test_accuracy)}

    def train(self, global_parameters: List[np.ndarray]) -> Tuple[float, float]:
        self.net.train()
        
        if self.model_name == 'snn':
            optimizer = optim.Adam(
                self.net.parameters(),
                lr=self.args.lr,
                weight_decay=getattr(self.args, 'weight_decay', 1e-4), 
                betas=(0.9, 0.999), 
                eps=1e-8
            )
        elif self.model_name in ['cnn', 'mlp']:
            optimizer = optim.SGD(
                self.net.parameters(),
                lr=self.args.lr,
                momentum=getattr(self.args, 'momentum', 0.9),
                weight_decay=getattr(self.args, 'weight_decay', 1e-4),
                nesterov=getattr(self.args, 'nesterov', False)
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")
        criterion = nn.CrossEntropyLoss()

        scheduler = None
        if getattr(self.args, 'use_lr_scheduler', False):
            try:
                scheduler = create_lr_scheduler(optimizer, self.args, len(self.trainloader))
            except Exception as e:
                print(f"Client {self.client_id}: Failed to create LR scheduler: {e}")

        self.optimizer = optimizer
        self.scheduler = scheduler

        epoch_losses = []
        correct = 0
        total = 0
        
        for epoch in range(int(getattr(self.args, 'local_ep', 1))):
            batch_losses = []
            if len(self.trainloader) == 0:
                print(f"Client {self.client_id}: No batches to train on, skipping epoch {epoch}")
                continue

            global_params = None
            use_fedprox = (self.args.strategy == 'fedprox' and getattr(self.args, 'fedprox_mu', 0.0) > 0)
            if use_fedprox:
                param_names = list(self.net.state_dict().keys())
                if len(global_parameters) != len(param_names):
                    raise ValueError(f"Parameter mismatch: expected {len(param_names)} global parameters, got {len(global_parameters)}")
                global_params = {
                    param_names[i]: torch.from_numpy(global_parameters[i]).to(self.device).detach()
                    for i in range(len(param_names))
                }

            for _, (data, targets) in enumerate(self.trainloader):
                data, targets = data.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                
                if self.model_name == 'snn':
                    _, mem_rec = self.net(data)
                    if isinstance(mem_rec, list):
                        mem_rec = torch.stack(mem_rec, dim=0)
                    elif not isinstance(mem_rec, torch.Tensor):
                        raise ValueError("Unexpected SNN output format: membrane potentials should be tensor")

                    outputs = mem_rec.sum(dim=0)
                else:
                    outputs = self.net(data)
                
                loss = criterion(outputs, targets)

                if use_fedprox:
                    proximal_term = torch.tensor(0.0, device=self.device)
                    for name, local_param in self.net.named_parameters():
                        if local_param.requires_grad and name in global_params:
  
                            proximal_term = proximal_term + torch.norm(local_param - global_params[name]) ** 2
                    loss = loss + (float(self.args.fedprox_mu) / 2.0) * proximal_term

                loss.backward()

                if self.model_name == 'snn':
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                    
                optimizer.step()

                if scheduler is not None and getattr(self.args, 'scheduler_step_per_batch', False):
                    try:
                        scheduler.step()
                    except Exception:
                        pass
            
                batch_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            if batch_losses:
                epoch_loss = sum(batch_losses) / len(batch_losses)
            else:
                epoch_loss = 0.0
            epoch_losses.append(epoch_loss)

            if scheduler is not None and not getattr(self.args, 'scheduler_step_per_batch', False):
                try:
                    scheduler.step()
                except Exception:
                    pass

            if use_fedprox and global_params is not None:
                del global_params
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
        else:
            avg_loss = 0.0
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def test(self) -> Tuple[float, float]:
        self.net.eval()
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        test_loss = 0
        
        if not hasattr(self, 'testloader') or len(self.testloader) == 0:
            return 0.0, 0.0
        with torch.no_grad():
            for data, targets in self.testloader:
                data, targets = data.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                if self.model_name == 'snn':
                    _, mem_rec = self.net(data)
                    if isinstance(mem_rec, list):
                        mem_rec = torch.stack(mem_rec, dim=0)
                    elif not isinstance(mem_rec, torch.Tensor):
                        raise ValueError("Unexpected SNN output format: membrane potentials should be tensor")

                    outputs = mem_rec.sum(dim=0)
                else:
                    outputs = self.net(data)

                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        num_batches = len(self.testloader) if len(self.testloader) > 0 else 0
        avg_loss = test_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss, accuracy
 
 


def client_fn(cid: str, train_data, test_data, client_data_dict, model_name: str, dataset: str, args) -> FlowerClient:
    """
    Create a Flower client with proper cleanup handling.
    
    This factory function creates a FlowerClient instance and ensures proper
    resource cleanup when the client is no longer needed.
    
    Args:
        cid: Client ID as string
        train_data: Training dataset
        test_data: Test dataset
        client_data_dict: Dictionary mapping client IDs to data indices
        model_name: Type of neural network model
        dataset: Dataset name
        args: Configuration arguments

    """
    client_id = int(cid)
    client_indices = client_data_dict.get(client_id)
    if client_indices is None:
        client_indices = client_data_dict.get(cid, set())
    if not client_indices:
        raise ValueError(f"Client {client_id} has no assigned data. Check data distribution.")

    local_bs = int(getattr(args, "local_bs", 32))
    min_samples = max(1, local_bs // 2)
    if len(client_indices) < min_samples:
        print(f"Warning: Client {client_id} has only {len(client_indices)} samples (minimum recommended: {min_samples})")

    try:
        client = FlowerClient(client_id, train_data, test_data, client_indices, model_name, dataset, args)

        weakref.finalize(client, FlowerClient.cleanup, client)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return client
    
    except Exception as e:
        print(f"Error creating client {client_id}: {e}")
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        raise


def get_federated_strategy(strategy_name: str, initial_parameters: List[np.ndarray], args) -> fl.server.strategy.Strategy:
    """
    Factory function to create different federated learning strategies.
    
    This function creates and configures various federated learning strategies
    with appropriate hyperparameters and configuration functions.
    
    Args:
        strategy_name: Name of the strategy to create
        initial_parameters: Initial model parameters
        args: Configuration arguments containing strategy parameters

    """
    def fit_config(server_round: int) -> Dict[str, Any]:
        """Configuration function for fit rounds."""
        config = {
            "server_round": server_round,
            "local_epochs": args.local_ep,
        }
        return config

    def evaluate_config(server_round: int) -> Dict[str, Any]:
        """Configuration function for evaluate rounds."""
        return {"server_round": server_round}


    frac = float(getattr(args, "frac", 1.0))
    num_users = int(getattr(args, "num_users", 1))
    min_fit = max(1, int(max(1, round(frac * num_users))))
    min_avail = max(min_fit, int(getattr(args, "min_available_clients", min_fit)))
    base_config = {
        "fraction_fit": frac,
        "fraction_evaluate": 1.0,
        "min_fit_clients": min_fit,
        "min_evaluate_clients": min(num_users, max(1, int(round(frac * num_users)))),
        "min_available_clients": min(num_users, min_avail),
        "on_fit_config_fn": fit_config,
        "on_evaluate_config_fn": evaluate_config,
        "fit_metrics_aggregation_fn": weighted_average,
        "evaluate_metrics_aggregation_fn": weighted_average,
    }

    if initial_parameters is not None:
        base_config["initial_parameters"] = fl.common.ndarrays_to_parameters(initial_parameters)

    class SaveFinalModelMixin:
        """
        Mixin class to save the final aggregated parameters.
        
        This mixin ensures that the final model parameters are preserved
        after the federated learning process completes.
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.final_parameters = None

        def aggregate_fit(
            self,
            server_round: int,
            results,
            failures,):
            """
            Aggregate results and save the final parameters.

            """
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(
                server_round, results, failures
            )
            if aggregated_parameters is not None:
                self.final_parameters = aggregated_parameters
            
            return aggregated_parameters, aggregated_metrics

    if strategy_name.lower() in ("fedavg", "fed_avg"):
        class FedAvgWithSave(SaveFinalModelMixin, fl.server.strategy.FedAvg):
            pass
        return FedAvgWithSave(**base_config)
        
    elif strategy_name.lower() == "fedprox":
        class FedProxWithSave(SaveFinalModelMixin, FedProx):
            pass
        mu = float(getattr(args, "fedprox_mu", 0.1))
        return FedProxWithSave(proximal_mu=mu, **base_config)

    elif strategy_name.lower() in ("fedadagrad", "fed_adagrad"):
        class FedAdagradWithSave(SaveFinalModelMixin, FedAdagrad):
            pass
        return FedAdagradWithSave(
            eta=float(getattr(args, "server_lr", 0.01)),
            eta_l=float(getattr(args, "client_lr", 0.01)),
            tau=float(getattr(args, "server_tau", 1e-9)),
            **base_config,
        )

    elif strategy_name.lower() in ("fedadam", "fed_adam"):
        class FedAdamWithSave(SaveFinalModelMixin, FedAdam):
            pass
        return FedAdamWithSave(
            eta=float(getattr(args, "server_lr", 0.01)),
            beta_1=float(getattr(args, "beta1", 0.9)),
            beta_2=float(getattr(args, "beta2", 0.999)),
            tau=float(getattr(args, "server_tau", 1e-9)),
            **base_config,
        )

    else:
        raise ValueError(f"Strategy {strategy_name} not supported")