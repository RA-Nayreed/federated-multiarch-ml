"""Fixed Flower-based Federated Learning Implementation for MNIST and CIFAR-10
Implements various strategies using the Flower framework with support for SNN, CNN, and MLP models
"""

import flwr as fl
import torch
import numpy as np
import argparse
import random

# Import from modules
from models import get_model
from utils import (get_parameters, set_parameters, save_model,
                   load_data, distribute_data)
from strategy import get_federated_strategy, client_fn 

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def run_simulation(args):
    """Run Flower simulation."""
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print("-" * 60)
    print(f"Starting Flower Federated Learning Simulation")
    print(f"Using device: {device}")
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
    def create_client_fn(context):
        cid = context.node_config["partition-id"]
        client = client_fn(str(cid), train_data, test_data, client_data_dict, args.model, args.dataset, args)
        return client

    # Determine client resources
    client_resources = {"num_cpus": 1, "num_gpus": 0.0} 
    if args.gpu and torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        if available_gpus > 0:
            # Assign a fraction of a GPU per client if needed, or 1 GPU if only one client fits
            # Fractional GPU assignment example (adjust as needed):
            # Aim for at least one client per GPU, or share if more clients than GPUs
            gpus_per_client = min(1.0, max(0.1, 1.0 / max(1, args.num_users)))
            # Ensure total requested doesn't exceed available (Ray handles this, but good to be aware)
            client_resources["num_gpus"] = gpus_per_client
            print(f"Assigned approx. {gpus_per_client} GPU(s) per client.")


    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=create_client_fn,
        num_clients=args.num_users,
        config=fl.server.ServerConfig(num_rounds=args.epochs),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={"ignore_reinit_error": True, "include_dashboard": False}, # Disable dashboard for simpler logs
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
            print(f"Warning: Could not set final model parameters: {e}. Saving initial model state.")
    else:
        print("Warning: Strategy did not provide final model parameters. Saving initial model state.")

    # Save the model
    save_model(final_model, args.model, args.dataset, args.strategy, args)
    return history


def main():
    parser = argparse.ArgumentParser(description='Flower Federated Learning with SNN/CNN/MLP')
    # Federated learning parameters
    parser.add_argument('--epochs', type=int, default=10, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1.0, help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    # Learning Rate Scheduler Arguments
    parser.add_argument('--use_lr_scheduler', action='store_true', help='Enable learning rate scheduler (warmup + cosine decay)')
    parser.add_argument('--warmup_epochs', type=float, default=5.0, help='Number of warmup epochs (can be fractional)')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='Minimum learning rate for cosine decay')
    
    # Model and dataset parameters
    parser.add_argument('--model', type=str, default='snn', choices=['mlp', 'cnn', 'snn'], help="model")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help="dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--gpu', action='store_true', help='use gpu')
    # SNN specific parameters
    parser.add_argument('--snn_timesteps', type=int, default=25, help='number of time steps for SNN')
    # Strategy parameter
    parser.add_argument('--strategy', type=str, default='fedavg',
                       choices=['fedavg', 'fedprox', 'fedadagrad', 'fedadam', 'feddyn'],
                       help='federated learning strategy')
    #Strategy-specific hyperparameters
    parser.add_argument('--fedprox_mu', type=float, default=0.1, help='FedProx proximal term coefficient')
    parser.add_argument('--feddyn_alpha', type=float, default=0.01, help='FedDyn alpha coefficient')
    
    args = parser.parse_args()

    # Strategy recommendation based on data distribution
    original_strategy = args.strategy
    if args.iid:
        if args.strategy == 'fedavg':
            print("Using FedAvg for IID data distribution")
        else:
            print(f"Note: {args.strategy.upper()} may be suboptimal for IID data. Consider FedAvg.")
    else: # Non-IID
        recommended_strategies = ['fedprox', 'feddyn']
        if args.strategy == 'fedavg':
            print("Warning: FedAvg may be suboptimal for non-IID data.")
            print(f"Consider using: {', '.join(recommended_strategies)} for better performance.")
            # Only auto-change if user explicitly wants it (removed auto-change for clarity)
            if input("Auto-switch to FedProx? (y/n): ").lower() == 'y':
                 args.strategy = 'fedprox'
                 print("Switched to FedProx for non-IID data distribution")
        elif args.strategy in recommended_strategies:
            print(f"Using {args.strategy.upper()} for non-IID data distribution")
        else:
             print(f"Using {args.strategy.upper()} for non-IID data distribution (ensure it's suitable).")


    # Adjust learning rates based on model type *only if default LR is used*
    default_lr = 0.01 
    if args.lr == default_lr:  
        if args.model == 'snn':
            args.lr = 0.01 
        elif args.model == 'cnn':
            args.lr = 0.01 
        elif args.model == 'mlp':
            if args.dataset == 'cifar10':
                args.lr = 0.01 
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
            # Get the last recorded metric for each type
            final_metrics = {}
            for metric_name, metric_values in history.metrics_centralized.items():
                if metric_values:
                    final_metrics[metric_name] = metric_values[-1]

            if 'test_accuracy' in final_metrics:
                # Metric value is a tuple (round_num, value)
                print(f"Final Centralized Test Accuracy: {final_metrics['test_accuracy'][1]:.2f}%")
            if 'train_accuracy' in final_metrics:
                print(f"Final Centralized Training Accuracy: {final_metrics['train_accuracy'][1]:.2f}%")
            # Add other metrics as needed (loss etc.)
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