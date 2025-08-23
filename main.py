"""
Flower-based Federated Learning Implementation for MNIST and CIFAR-10.

This module implements federated learning using the Flower framework with support
for various neural network architectures (SNN, CNN, MLP) and federated learning
strategies (FedAvg, FedProx, FedAdagrad, FedAdam).

"""

import flwr as fl
import torch
import numpy as np
import argparse
import random
from typing import Optional

# Import from modules
from models import get_model
from utils import (get_parameters, set_parameters, save_model,
                   load_data, distribute_data)
from strategy import get_federated_strategy, client_fn 

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def run_simulation(args: argparse.Namespace) -> Optional[fl.server.history.History]:
    """
    Run Flower federated learning simulation.
    
    This function orchestrates the complete federated learning process:
    1. Data loading and distribution to clients
    2. Model initialization and parameter setup
    3. Strategy configuration and client resource allocation
    4. Simulation execution with Flower
    5. Model saving and result reporting

    """
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    
    print("-" * 60)
    print("Starting Flower Federated Learning Simulation")
    print(f"Using device: {device}")
    if device.type == "cuda":
        # GPU info
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device index: {torch.cuda.current_device()}")

    print(f"Dataset: {args.dataset.upper()} |  Model: {args.model.upper()}")
    print(f"Strategy: {args.strategy.upper()}")
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
    initial_model.to(device)
    initial_parameters = get_parameters(initial_model)

    # Create federated strategy
    strategy = get_federated_strategy(args.strategy, initial_parameters, args)

    # Define client function for simulation
    def create_client_fn(cid: str):
        """Create a client instance for the given client ID."""
        client = client_fn(str(cid), train_data, test_data, client_data_dict, 
                          args.model, args.dataset, args)
        return client

    # Determine client resources
    client_resources = {"num_cpus": 1, "num_gpus": 0.0} 
    if args.gpu and torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        if available_gpus > 0:
            # Simple allocation: distribute GPUs evenly among clients
            gpus_per_client = min(1.0, available_gpus / args.num_users)
            client_resources["num_gpus"] = gpus_per_client
            print(f"Assigned {gpus_per_client:.2f} GPU(s) per client "
                  f"({available_gpus} total GPUs, {args.num_users} clients)")

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=create_client_fn,
        num_clients=args.num_users,
        config=fl.server.ServerConfig(num_rounds=args.epochs),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={"ignore_reinit_error": True, "include_dashboard": False},
    )

    # Get final model parameters from the strategy
    final_model = get_model(args.model, args.dataset, args.snn_timesteps)
    final_model.to(device)
    if hasattr(strategy, 'final_parameters') and strategy.final_parameters is not None:
        try:
            final_parameters = fl.common.parameters_to_ndarrays(strategy.final_parameters)
            set_parameters(final_model, final_parameters)
            print("Successfully retrieved final model parameters from federated training.")
        except Exception as e:
            print(f"Warning: Could not set final model parameters: {e}. "
                  f"Saving initial model state.")
    else:
        print("Warning: Strategy did not provide final model parameters. "
              "Saving initial model state.")

    # Save the model
    save_model(final_model, args.model, args.dataset, args.strategy, args)
    return history


def main():
    """
    Main entry point for the federated learning application.
    
    This function:
    1. Parses command line arguments
    2. Validates and adjusts configuration based on model/dataset
    3. Provides strategy recommendations for data distributions
    4. Executes the federated learning simulation
    5. Reports final results and metrics
    """
    parser = argparse.ArgumentParser(description='Flower Federated Learning with SNN/CNN/MLP')
    
    # Federated learning parameters
    parser.add_argument('--epochs', type=int, default=40, 
                       help="Number of federated learning rounds")
    parser.add_argument('--num_users', type=int, default=10, 
                       help="Number of clients participating in training")
    parser.add_argument('--frac', type=float, default=1.0, 
                       help='Fraction of clients to sample per round')
    parser.add_argument('--local_ep', type=int, default=6, 
                       help="Number of local training epochs per client")
    parser.add_argument('--local_bs', type=int, default=32, 
                       help="Local batch size for client training")
    parser.add_argument('--lr', type=float, default=0.01, 
                       help='Learning rate for client training')
    
    # Learning Rate Scheduler Arguments
    parser.add_argument('--use_lr_scheduler', action='store_true', 
                       help='Enable learning rate scheduler (warmup + cosine decay)')
    parser.add_argument('--warmup_epochs', type=float, default=5.0, 
                       help='Number of warmup epochs (can be fractional)')
    parser.add_argument('--lr_min', type=float, default=1e-6, 
                       help='Minimum learning rate for cosine decay')
    
    # Model and dataset parameters
    parser.add_argument('--model', type=str, default='snn', 
                       choices=['mlp', 'cnn', 'snn'], help="Neural network architecture")
    parser.add_argument('--dataset', type=str, default='mnist', 
                       choices=['mnist', 'cifar10'], help="Dataset for training")
    parser.add_argument('--iid', action='store_true', 
                       help='Use IID data distribution (default: Non-IID)')
    parser.add_argument('--gpu', action='store_true', 
                       help='Use GPU acceleration if available')
    
    # SNN specific parameters
    parser.add_argument('--snn_timesteps', type=int, default=25, 
                       help='Number of time steps for SNN temporal dynamics')
    
    # Strategy parameter
    parser.add_argument('--strategy', type=str, default='fedavg',
                       choices=['fedavg', 'fedprox', 'fedadagrad', 'fedadam'],
                       help='Federated learning strategy')
    
    # Strategy-specific hyperparameters
    parser.add_argument('--fedprox_mu', type=float, default=0.1, 
                       help='FedProx proximal term coefficient')

    
    args = parser.parse_args()
    
    # Validate SNN timesteps
    if args.model == 'snn' and args.snn_timesteps <= 0:
        raise ValueError("SNN timesteps must be positive")
    
    # Validate strategy parameters
    if args.fedprox_mu < 0:
        raise ValueError("FedProx mu must be non-negative")


    # Strategy recommendation based on data distribution
    if args.iid:
        if args.strategy == 'fedavg':
            print("Using FedAvg for IID data distribution")
        else:
            print(f"Note: {args.strategy.upper()} may be suboptimal for IID data. Consider FedAvg.")
    else:  # Non-IID
        recommended_strategies = ['fedprox']
        if args.strategy == 'fedavg':
            print("Warning: FedAvg may be suboptimal for non-IID data.")
            print(f"Consider using: {', '.join(recommended_strategies)} for better performance.")

        elif args.strategy in recommended_strategies:
            print(f"Using {args.strategy.upper()} for non-IID data distribution")
        else:
             print(f"Using {args.strategy.upper()} for non-IID data distribution "
                   f"(ensure it's suitable).")

    # Adjust learning rates based on model type if default LR is used
    default_lr = 0.01 
    if args.lr == default_lr:  
        if args.model == 'snn':
            args.lr = 0.005 
        elif args.model == 'cnn':
            args.lr = 0.01 
        elif args.model == 'mlp':
            if args.dataset == 'cifar10':
                args.lr = 0.005 
            else:
                args.lr = 0.01 
        print(f"Auto-adjusted learning rate to {args.lr} for {args.model} model")
    else:
        print(f"Using user-specified learning rate: {args.lr}")
    print(f"Configuration: {vars(args)}")

    # Run simulation
    try:
        history = run_simulation(args)
        print("\nSimulation completed successfully!")

        # Print final results if available
        if history and hasattr(history, 'metrics_centralized') and history.metrics_centralized:
            final_metrics = {}
            for metric_name, metric_values in history.metrics_centralized.items():
                if metric_values:
                    final_metrics[metric_name] = metric_values[-1]

            if 'test_accuracy' in final_metrics:
                print(f"Final Centralized Test Accuracy: {final_metrics['test_accuracy'][1]:.2f}%")
            if 'train_accuracy' in final_metrics:
                print(f"Final Centralized Training Accuracy: {final_metrics['train_accuracy'][1]:.2f}%")
        elif history:
            print("Simulation finished, but no centralized metrics were recorded.")
        else:
            print("Simulation finished, but history object is None.")

    except Exception as e:
        print(f"An error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()