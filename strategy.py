"""
Flower Client and Strategy Implementations for Federated Learning.

"""

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple, Optional, Any
from flwr.server.strategy import FedProx, FedAdagrad, FedAdam
from models import get_model
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
        self.client_indices = client_indices
        self.model_name = model_name
        self.dataset = dataset
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
        self.net = get_model(model_name, dataset, args.snn_timesteps)
        self.net.to(self.device)
        
        client_dataset = Subset(train_data, list(client_indices))
        effective_batch_size = min(args.local_bs, len(client_dataset))
        if effective_batch_size != args.local_bs:
            print(f"Client {client_id}: Reduced batch size from {args.local_bs} to {effective_batch_size} due to small dataset")

        self.trainloader = DataLoader(client_dataset, batch_size=effective_batch_size, shuffle=True,
                                    pin_memory=True if self.device.type == 'cuda' else False)
        self.testloader = DataLoader(test_data, batch_size=128, shuffle=False,
                                     pin_memory=True if self.device.type == 'cuda' else False)
        print(f"Client {client_id}: {len(client_indices)} samples on device {self.device}")

    def cleanup(self):
        """
        Clean up resources to prevent memory leaks.
        
        This method:
        - Removes DataLoaders
        - Moves model to CPU and deletes it
        - Forces garbage collection
        - Clears GPU cache if using CUDA
        """
        if hasattr(self, 'trainloader'):
            del self.trainloader
        if hasattr(self, 'testloader'):
            del self.testloader
        
        if hasattr(self, 'net'):
            self.net.cpu() 
            del self.net
        
        import gc
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Return current local model parameters.
        
        """
        return get_parameters(self.net)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """
        Train the model on the locally held training set.
        
        Args:
            parameters: Global model parameters from server
            config: Configuration dictionary containing training parameters
            
        Returns:
            Tuple containing:
            - Updated model parameters
            - Number of training examples
            - Training metrics (loss, accuracy)
        """
        set_parameters(self.net, parameters)
        train_loss, train_accuracy = self.train(parameters)
        return (
            get_parameters(self.net),
            len(self.trainloader.dataset),
            {"train_loss": float(train_loss), "train_accuracy": float(train_accuracy)}
        )

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, float]]:
        """
        Evaluate the model on the locally held test set.
        
        """
        set_parameters(self.net, parameters)
        test_loss, test_accuracy = self.test()
        return float(test_loss), len(self.testloader.dataset), {"test_accuracy": float(test_accuracy)}

    def train(self, global_parameters: List[np.ndarray]) -> Tuple[float, float]:
        """
        Train the model locally for the specified number of epochs.
        
        This method handles:
        - Optimizer selection based on model type
        - Learning rate scheduling
        - FedProx proximal term calculation
        - Gradient clipping
        - Training metrics computation

        """
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

        epoch_losses = []
        correct = 0
        total = 0
        
        for epoch in range(self.args.local_ep):
            batch_losses = []
            if len(self.trainloader) == 0:
                print(f"Client {self.client_id}: No batches to train on, skipping epoch {epoch}")
                continue
                
            for batch_idx, (data, targets) in enumerate(self.trainloader):
                data, targets = data.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                
                if self.model_name == 'snn':
                    _, mem_rec = self.net(data)
                    if isinstance(mem_rec, list):
                        mem_rec = torch.stack(mem_rec, dim=0)
                    elif not isinstance(mem_rec, torch.Tensor):
                        raise ValueError("Unexpected SNN output format: membrane potentials should be tensor")
                    # Use sum across time dimension for consistent behavior
                    outputs = mem_rec.sum(dim=0)
                else:
                    outputs = self.net(data)
                
                loss = criterion(outputs, targets)

                if self.args.strategy == 'fedprox' and getattr(self.args, 'fedprox_mu', 0.1) > 0:
                    param_names = list(self.net.state_dict().keys())
                    if len(global_parameters) != len(param_names):
                        raise ValueError(f"Parameter mismatch: expected {len(param_names)} global parameters, got {len(global_parameters)}")
                    global_params = {
                        param_names[i]: torch.from_numpy(global_parameters[i]).to(self.device)
                        for i in range(len(param_names))
                    }

                    proximal_term = torch.tensor(0.0, device=self.device)
                    for name, local_param in self.net.named_parameters():
                        if local_param.requires_grad and name in global_params:
                            proximal_term += torch.norm(local_param - global_params[name]) ** 2
                    
                    loss = loss + (self.args.fedprox_mu / 2.0) * proximal_term
                    
                    for t in global_params.values():
                        del t
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

                loss.backward()

                if self.model_name == 'snn':
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                    
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()
            
                batch_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            if batch_losses:
                epoch_loss = sum(batch_losses) / len(batch_losses)
            else:
                epoch_loss = 0.0
            epoch_losses.append(epoch_loss)

        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
        else:
            avg_loss = 0.0
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def test(self) -> Tuple[float, float]:
        """
        Test the model on the local test dataset.

        """
        self.net.eval()
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        test_loss = 0
        
        with torch.no_grad():
            for data, targets in self.testloader:
                data, targets = data.to(self.device), targets.to(self.device)
                if self.model_name == 'snn':
                    _, mem_rec = self.net(data)
                    if isinstance(mem_rec, list):
                        mem_rec = torch.stack(mem_rec, dim=0)
                    elif not isinstance(mem_rec, torch.Tensor):
                        raise ValueError("Unexpected SNN output format: membrane potentials should be tensor")
                    # Use sum across time dimension for consistent behavior
                    outputs = mem_rec.sum(dim=0)
                else:
                    outputs = self.net(data)

                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = test_loss / len(self.testloader) if len(self.testloader) > 0 else 0.0
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
    client_indices = client_data_dict.get(client_id, set())
    if not client_indices:
        raise ValueError(f"Client {client_id} has no assigned data. Check data distribution.")

    min_samples = max(1, args.local_bs // 2)
    if len(client_indices) < min_samples:
        print(f"Warning: Client {client_id} has only {len(client_indices)} samples (minimum recommended: {min_samples})")

    try:
        client = FlowerClient(client_id, train_data, test_data, client_indices, model_name, dataset, args)
        
        def cleanup_proxy():
            if hasattr(client, 'cleanup'):
                client.cleanup()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        import weakref
        weakref.finalize(client, cleanup_proxy)
        
        return client
    except Exception as e:
        print(f"Error creating client {client_id}: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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

    base_config = {
        "fraction_fit": args.frac,
        "fraction_evaluate": 1.0,
        "min_fit_clients": max(1, int(args.frac * args.num_users)),
        "min_evaluate_clients": args.num_users,
        "min_available_clients": args.num_users,
        "initial_parameters": fl.common.ndarrays_to_parameters(initial_parameters),
        "on_fit_config_fn": fit_config,
        "on_evaluate_config_fn": evaluate_config,
        "fit_metrics_aggregation_fn": weighted_average,
        "evaluate_metrics_aggregation_fn": weighted_average,
    }

    class SaveFinalModelMixin:
        """
        Mixin class to save the final aggregated parameters.
        
        This mixin ensures that the final model parameters are preserved
        after the federated learning process completes.
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.final_parameters = None

        def aggregate_fit(self, server_round: int, results: List[fl.server.client_proxy.ClientProxy], 
                         failures: List[fl.server.client_proxy.ClientProxy]) -> Tuple[Optional[fl.common.Parameters], Dict[str, Any]]:
            """
            Aggregate results and save the final parameters.

            """
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(
                server_round, results, failures
            )
            if aggregated_parameters is not None:
                self.final_parameters = aggregated_parameters
            
            return aggregated_parameters, aggregated_metrics

    if strategy_name == "fedavg":
        class FedAvgWithSave(SaveFinalModelMixin, fl.server.strategy.FedAvg):
            pass
        return FedAvgWithSave(**base_config)
        
    elif strategy_name == "fedprox":
        class FedProxWithSave(SaveFinalModelMixin, FedProx):
            pass
        mu = getattr(args, 'fedprox_mu', 0.1)
        return FedProxWithSave(proximal_mu=mu, **base_config)

    elif strategy_name == "fedadagrad":
        class FedAdagradWithSave(SaveFinalModelMixin, FedAdagrad):
            pass
        return FedAdagradWithSave(eta=0.01, eta_l=0.01, tau=1e-9, **base_config)

    elif strategy_name == "fedadam":
        class FedAdamWithSave(SaveFinalModelMixin, FedAdam):
            pass
        return FedAdamWithSave(eta=0.01, beta_1=0.9, beta_2=0.999, tau=1e-9, **base_config)

    elif strategy_name == "feddyn":
        class SaveModelFedDyn(fl.server.strategy.Strategy):
            """
            FedDyn strategy implementation with final model saving.
            
            FedDyn (Federated Dynamic regularization) adds a dynamic regularization
            term to prevent client drift in federated learning.
            
            Args:
                fraction_fit: Fraction of clients to sample for training
                fraction_evaluate: Fraction of clients to sample for evaluation
                min_fit_clients: Minimum number of clients for training
                min_evaluate_clients: Minimum number of clients for evaluation
                min_available_clients: Minimum number of available clients
                initial_parameters: Initial model parameters
                on_fit_config_fn: Function to configure fit rounds
                on_evaluate_config_fn: Function to configure evaluate rounds
                fit_metrics_aggregation_fn: Function to aggregate fit metrics
                evaluate_metrics_aggregation_fn: Function to aggregate evaluate metrics
                alpha: FedDyn regularization parameter
            """
            def __init__(self, fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=2,
                         min_evaluate_clients=2, min_available_clients=2, initial_parameters=None,
                         on_fit_config_fn=None, on_evaluate_config_fn=None,
                         fit_metrics_aggregation_fn=None, evaluate_metrics_aggregation_fn=None,
                         alpha=0.01):
                super().__init__()
                self.fraction_fit = fraction_fit
                self.fraction_evaluate = fraction_evaluate
                self.min_fit_clients = min_fit_clients
                self.min_evaluate_clients = min_evaluate_clients
                self.min_available_clients = min_available_clients
                
                if initial_parameters is None:
                    raise ValueError("FedDyn requires initial_parameters to be provided")
                self.initial_parameters = initial_parameters
                self.server_state = fl.common.parameters_to_ndarrays(initial_parameters)
                self.h = [np.zeros_like(param) for param in self.server_state]
                
                self.on_fit_config_fn = on_fit_config_fn
                self.on_evaluate_config_fn = on_evaluate_config_fn
                self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
                self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
                if alpha <= 0:
                    raise ValueError(f"FedDyn requires alpha > 0, got {alpha}")
                self.alpha = alpha
                self.final_parameters = None

            def initialize_parameters(self, client_manager) -> fl.common.Parameters:
                """Initialize model parameters."""
                return self.initial_parameters

            def configure_fit(self, server_round: int, parameters: fl.common.Parameters, 
                            client_manager) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
                """Configure the next round of training."""
                sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
                clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
                config = {}
                if self.on_fit_config_fn is not None:
                    config = self.on_fit_config_fn(server_round)
                config["alpha"] = self.alpha
                return [(client, fl.common.FitIns(parameters, config)) for client in clients]

            def aggregate_fit(self, server_round: int, results: List[fl.server.client_proxy.ClientProxy], 
                            failures: List[fl.server.client_proxy.ClientProxy]) -> Tuple[Optional[fl.common.Parameters], Dict[str, Any]]:
                """
                Aggregate fit results using FedDyn algorithm.
                
                This method implements the FedDyn aggregation which includes:
                1. Computing average client parameters
                2. Updating the dynamic regularization term h
                3. Computing new server state with regularization
                """
                if not results:
                    return None, {}
                weights_results = [
                    (fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                    for _, fit_res in results
                ]
                if self.server_state is None:
                    self.server_state = fl.common.parameters_to_ndarrays(self.initial_parameters)
                    self.h = [np.zeros_like(param) for param in self.server_state]

                total_examples = sum([num_examples for _, num_examples in weights_results])
                if total_examples == 0:
                    return fl.common.ndarrays_to_parameters(self.server_state), {}

                avg_client_params = [
                    np.sum([weights[i] * num_examples for weights, num_examples in weights_results], axis=0) / total_examples
                    for i in range(len(weights_results[0][0]))
                ]
                
                new_h = []
                for h_param, server_param, avg_param in zip(self.h, self.server_state, avg_client_params):
                    avg_update = avg_param - server_param
                    new_h_param = h_param + avg_update
                    new_h.append(new_h_param)
                self.h = new_h

                new_server_state = []
                for h_param, avg_param in zip(self.h, avg_client_params):
                    new_param = avg_param - (1.0 / self.alpha) * h_param
                    new_server_state.append(new_param)

                self.server_state = new_server_state
                
                aggregated_parameters = fl.common.ndarrays_to_parameters(self.server_state)
                self.final_parameters = aggregated_parameters
                metrics_aggregated = {}
                if self.fit_metrics_aggregation_fn:
                    fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                    metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
                return aggregated_parameters, metrics_aggregated

            def configure_evaluate(self, server_round: int, parameters: fl.common.Parameters, 
                                 client_manager) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
                """Configure the next round of evaluation."""
                if self.fraction_evaluate == 0.0:
                    return []
                sample_size, min_num_clients = self.num_evaluate_clients(client_manager.num_available())
                clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
                config = {}
                if self.on_evaluate_config_fn is not None:
                    config = self.on_evaluate_config_fn(server_round)
                return [(client, fl.common.EvaluateIns(parameters, config)) for client in clients]

            def aggregate_evaluate(self, server_round: int, results: List[fl.server.client_proxy.ClientProxy], 
                                 failures: List[fl.server.client_proxy.ClientProxy]) -> Tuple[Optional[fl.common.Parameters], Dict[str, Any]]:
                """Aggregate evaluation results."""
                if not results:
                    return None, {}
                metrics_aggregated = {}
                if self.evaluate_metrics_aggregation_fn:
                    eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
                    metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
                return None, metrics_aggregated

            def evaluate(self, server_round: int, parameters: fl.common.Parameters) -> Optional[Tuple[float, Dict[str, Any]]]:
                """Evaluate the current global model."""
                return None

            def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
                """Determine number of clients for training."""
                num_clients = int(num_available_clients * self.fraction_fit)
                return max(num_clients, self.min_fit_clients), self.min_fit_clients

            def num_evaluate_clients(self, num_available_clients: int) -> Tuple[int, int]:
                """Determine number of clients for evaluation."""
                num_clients = int(num_available_clients * self.fraction_evaluate)
                return max(num_clients, self.min_evaluate_clients), self.min_evaluate_clients

        alpha = getattr(args, 'feddyn_alpha', 0.01) 
        return SaveModelFedDyn(alpha=alpha, **base_config)

    else:
        raise ValueError(f"Strategy {strategy_name} not supported")
