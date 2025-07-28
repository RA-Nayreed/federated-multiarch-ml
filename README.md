# Federated Multiarch Machine Learning
A Flower-based federated learning framework supporting MLP, CNN and Spiking Neural Networks (SNN) under both IID and non‑IID data distributions.

## Features

- **Multiple Model Types**: Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), and Spiking Neural Network (SNN)
- **Datasets**: MNIST and CIFAR-10 support
- **Distribution Options**: Both IID and Non-IID data distribution
- **Easy Training**: Interactive launcher with predefined configurations
- **Model Inference**: Simple inference script for saved models
- **GPU Support**: CUDA acceleration when available

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/RA-Nayreed/federated-multiarch-ml.git
cd federated-multiarch-ml

# Install dependencies
pip install -r requirements.txt
```

### Run Training

#### Option 1: Interactive Launcher (Recommended)
```bash
python training.py
```
Choose from 12 predefined configurations covering all model-dataset combinations.

#### Option 2: Direct Command
```bash
# Example: SNN on MNIST with 10 clients
python fed_learning.py --model snn --dataset mnist --epochs 15 --num_users 10 --iid --gpu

# Example: CNN on CIFAR-10 with Non-IID distribution
python fed_learning.py --model cnn --dataset cifar10 --epochs 20 --num_users 16 --gpu
```

### Run Inference

```bash
# Single image prediction
python inference.py --model_path models/snn_mnist_clients_rounds.pth --image_path test_image.png

# Batch inference on folder
python inference.py --model_path models/cnn_cifar10_clients_rounds.pth --image_folder test_images/
```

## Model Architectures

### Multi-Layer Perceptron (MLP)
- 4-layer fully connected network
- ReLU activation with dropout
- Suitable for both MNIST and CIFAR-10

### Convolutional Neural Network (CNN)
- **MNIST**: 3 conv layers + 2 FC layers with batch normalization
- **CIFAR-10**: Enhanced architecture with 3 conv layers + 3 FC layers

### Spiking Neural Network (SNN)
- **MNIST**: 3-layer fully connected SNN with Leaky Integrate-and-Fire neurons
- **CIFAR-10**: Convolutional SNN with temporal dynamics
- Configurable time steps (default: 25-30)

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 10 | Number of federated rounds |
| `--num_users` | 10 | Number of clients |
| `--frac` | 1.0 | Fraction of clients per round |
| `--local_ep` | 5 | Local training epochs |
| `--local_bs` | 32 | Local batch size |
| `--lr` | 0.01 | Learning rate (auto-adjusted for SNNs) |
| `--model` | cnn | Model type: mlp, cnn, snn |
| `--dataset` | mnist | Dataset: mnist, cifar10 |
| `--iid` | False | Use IID data distribution |
| `--gpu` | False | Enable GPU acceleration |
| `--snn_timesteps` | 25 | SNN simulation time steps |

## Project Structure

```
federated-multiarch-ml/
├── fed_learning.py      # Main federated learning implementation
├── inference.py         # Model inference script
├── training.py          # Interactive training launcher
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── models/             # Saved models (created automatically)
└── data/               # Dataset cache (created automatically)
```



## Requirements

- Python 3.8+
- PyTorch 1.9+
- Flower 1.0+
- SNNTorch (for SNN models)
- See `requirements.txt` for complete list


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Flower Framework](https://flower.dev/) for federated learning infrastructure
- [SNNTorch](https://snntorch.readthedocs.io/) for spiking neural network implementation
- PyTorch team for the deep learning framework
