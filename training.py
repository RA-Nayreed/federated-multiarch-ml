"""
Interactive Federated Learning Launcher

Launches predefined FL setups using fed_learning.py main().
"""

import sys
import os


from main import main as flower_main  # Change to actual name if needed

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

def run_example(example):
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




