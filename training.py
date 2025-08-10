"""
Interactive Federated Learning Launcher.

The launcher includes predefined configurations for:
- Different neural network architectures (MLP, CNN, SNN)
- Multiple datasets (MNIST, CIFAR-10)
- Both IID and Non-IID data distributions
- Various hyperparameter settings optimized for each model-dataset combination

"""

import sys
import os
from typing import Dict, Any
from main import main as flower_main 

# Predefined experiment configurations
EXAMPLES = [
    {
        "name": "MLP on MNIST (IID)",
        "args": ["--model", "mlp", "--dataset", "mnist", "--epochs", "40", "--num_users", "10", "--iid", "--gpu", "--local_ep", "6"]
    },
    {
        "name": "MLP on MNIST (Non-IID)",
        "args": ["--model", "mlp", "--dataset", "mnist", "--epochs", "40", "--num_users", "10", "--gpu", "--local_ep", "6"]
    },
    {
        "name": "MLP on CIFAR-10 (IID)",
        "args": ["--model", "mlp", "--dataset", "cifar10", "--epochs", "40", "--num_users", "16", "--iid", "--gpu", "--local_ep", "6"]
    },
    {
        "name": "MLP on CIFAR-10 (Non-IID)",
        "args": ["--model", "mlp", "--dataset", "cifar10", "--epochs", "40", "--num_users", "16", "--gpu", "--local_ep", "6"]
    },
    {
        "name": "CNN on MNIST (IID)",
        "args": ["--model", "cnn", "--dataset", "mnist", "--epochs", "40", "--num_users", "10", "--iid", "--gpu", "--local_ep", "6"]
    },
    {
        "name": "CNN on MNIST (Non-IID)",
        "args": ["--model", "cnn", "--dataset", "mnist", "--epochs", "40", "--num_users", "10", "--gpu", "--local_ep", "6"]
    },
    {
        "name": "CNN on CIFAR-10 (IID)",
        "args": ["--model", "cnn", "--dataset", "cifar10", "--epochs", "40", "--num_users", "16", "--iid", "--gpu", "--local_ep", "6"]
    },
    {
        "name": "CNN on CIFAR-10 (Non-IID)",
        "args": ["--model", "cnn", "--dataset", "cifar10", "--epochs", "40", "--num_users", "16", "--gpu", "--local_ep", "6"]
    },
    {
        "name": "SNN on MNIST (IID)",
        "args": ["--model", "snn", "--dataset", "mnist", "--epochs", "40", "--num_users", "10", "--iid", "--local_ep", "6", "--gpu", "--snn_timesteps", "25"]
    },
    {
        "name": "SNN on MNIST (Non-IID)",
        "args": ["--model", "snn", "--dataset", "mnist", "--epochs", "40", "--num_users", "10", "--local_ep", "6", "--gpu", "--snn_timesteps", "25"]
    },
    {
        "name": "SNN on CIFAR-10 (IID)",
        "args": ["--model", "snn", "--dataset", "cifar10", "--epochs", "40", "--num_users", "16", "--iid", "--local_ep", "6", "--gpu", "--snn_timesteps", "30"]
    },
    {
        "name": "SNN on CIFAR-10 (Non-IID)",
        "args": ["--model", "snn", "--dataset", "cifar10", "--epochs", "40", "--num_users", "16", "--local_ep", "6", "--gpu", "--snn_timesteps", "30"]
    }
]


def run_example(example: Dict[str, Any]) -> bool:
    """
    Execute a single federated learning experiment.
    
    This function runs a predefined experiment configuration by:
    1. Setting up the command-line arguments
    2. Calling the main federated learning function
    3. Handling any exceptions and reporting results
    
    Args:
        example: Dictionary containing experiment configuration with 'name' and 'args' keys
        
    Returns:
        bool: True if experiment completed successfully, False otherwise
        
    Example:
        >>> example = {"name": "Test", "args": ["--model", "mlp", "--dataset", "mnist"]}
        >>> success = run_example(example)
    """
    print("\n" + "="*60)
    print(f"Running: {example['name']}")
    print("="*60)
    
    sys.argv = [sys.argv[0]] + example["args"]
    try:
        flower_main()
        print(f"\n{example['name']} Completed successfully.")
        return True
    except Exception as e:
        print(f"\nError in {example['name']}: {e}")
        return False


def main():
    """
    Main entry point for the interactive federated learning launcher.
    
    This function:
    1. Changes to the script directory for consistent file paths
    2. Displays available experiment configurations
    3. Prompts user for experiment selection
    4. Executes selected experiments
    5. Reports summary statistics
    
    The user can choose to run:
    - A single experiment by entering its number
    - All experiments by entering 'all'
    
    After completion, trained models are saved in the ./models/ directory.
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("Federated Learning Launcher")
    print("Available configurations:")
    for idx, ex in enumerate(EXAMPLES, 1):
        print(f"{idx}. {ex['name']}")

    while True:
        choice = input(f"\nEnter example number (1-{len(EXAMPLES)}) or 'all': ").strip().lower()
        if choice == "all":
            selections = EXAMPLES
            break
        if choice.isdigit():
            i = int(choice) - 1
            if 0 <= i < len(EXAMPLES):
                selections = [EXAMPLES[i]]
                break
        print("Invalid choice; try again.")

    successes = 0
    for ex in selections:
        if run_example(ex):
            successes += 1

    print("\n" + "="*60)
    print(f"Summary: {successes}/{len(selections)} runs succeeded.")
    print("Models saved in ./models/")
    print("="*60)


if __name__ == "__main__":
    main()




