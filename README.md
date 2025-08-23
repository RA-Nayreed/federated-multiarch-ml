# Federated Multiarch Machine Learning

A Flower-based federated learning framework supporting MLP, CNN and Spiking Neural Networks (SNN) under both IID and non-IID data distributions.

## Features

- **Multiple Model Types**: Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), and Spiking Neural Network (SNN)
- **Federated Strategies**: FedAvg, FedProx, FedAdagrad, FedAdam
- **Datasets**: MNIST and CIFAR-10 support
- **Distribution Options**: Both IID and Non-IID data distribution
- **Easy Training**: Interactive launcher with predefined configurations
- **Model Inference**: Simple inference script for saved models
- **GPU Support**: CUDA acceleration when available
- **Advanced Features**: Learning rate scheduling, gradient clipping, automatic strategy recommendations

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
python main.py --model snn --dataset mnist --epochs 15 --num_users 10 --iid --gpu

# Example: CNN on CIFAR-10 with Non-IID distribution
python main.py --model cnn --dataset cifar10 --epochs 20 --num_users 16 --gpu

# Example: Advanced training with FedProx strategy
python main.py --model cnn --dataset mnist --strategy fedprox --fedprox_mu 0.1 --epochs 30 --gpu
```

### Run Inference

```bash
# Single image prediction
python inference.py --model_path models/snn_mnist_fedavg_clients10_rounds15.pth --image_path test_image.png

# Batch inference on folder
python inference.py --model_path models/cnn_cifar10_fedavg_clients16_rounds20.pth --image_folder test_images/
```

## Model Architectures

### Multi-Layer Perceptron (MLP)
- 4-layer fully connected network (512→256→128→classes)
- ReLU activation with dropout regularization
- Suitable for both MNIST and CIFAR-10

### Convolutional Neural Network (CNN)
- **MNIST (SimpleCNN)**: 3 convolutional blocks (2 × Conv2d per block) each followed by BatchNorm + LeakyReLU, spatial down-sampling via stride = 2 in blocks 2 & 3, Dropout2d after every block, Global Average Pooling, then 1 hidden fully-connected layer (128 units) with dropout before the output layer.
- **CIFAR-10 (CIFAR10CNN)**: 3 convolutional blocks  
  • Block 1: Conv2d 3→64, Conv2d 64→128 (stride 2) with dropout 0.2  
  • Block 2: Conv2d 128→256, Conv2d 256→256 (stride 2) with dropout 0.2  
  • Block 3: Conv2d 256→512, Conv2d 512→256 (1×1) with dropout 0.3  
  Followed by Adaptive Average Pooling to (1,1) and two hidden fully-connected layers (256→192→128) with dropout 0.5 before the output layer.
- All convolutional layers use BatchNorm and LeakyReLU activations; no MaxPooling layers are used (down-sampling is achieved via stride).

### Spiking Neural Network (SNN)
- **MNIST**: 3-layer fully connected SNN with Leaky Integrate-and-Fire neurons
- **CIFAR-10**: Convolutional SNN with temporal dynamics
- Configurable time steps (default: 25-30)
- Surrogate gradient learning with membrane potential tracking

## Federated Learning Strategies

| Strategy | Best For | Description |
|----------|----------|-------------|
| `fedavg` | IID data | Standard federated averaging |
| `fedprox` | Non-IID data | Proximal regularization for heterogeneous clients |
| `fedadagrad` | Adaptive learning | Adaptive gradient-based optimization |
| `fedadam` | Complex optimization | Adam-based federated learning |


## Command Line Arguments

### Core Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 40 | Number of federated rounds |
| `--num_users` | 10 | Number of clients |
| `--frac` | 1.0 | Fraction of clients per round |
| `--local_ep` | 6 | Local training epochs |
| `--local_bs` | 32 | Local batch size |
| `--lr` | 0.01 | Learning rate (auto-adjusted per model) |
| `--model` | snn | Model type: mlp, cnn, snn |
| `--dataset` | mnist | Dataset: mnist, cifar10 |
| `--strategy` | fedavg | FL strategy: fedavg, fedprox, fedadagrad, fedadam |
| `--iid` | False | Use IID data distribution |
| `--gpu` | False | Enable GPU acceleration |

### Advanced Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--snn_timesteps` | 25 | SNN simulation time steps |
| `--fedprox_mu` | 0.1 | FedProx proximal term coefficient |

| `--use_lr_scheduler` | False | Enable learning rate scheduling |
| `--warmup_epochs` | 5.0 | Warmup epochs for LR scheduler |
| `--auto_switch_fedprox` | False | Auto-switch to FedProx for non-IID data |

## Project Structure

```
federated-multiarch-ml/
├── main.py              # Main federated learning implementation
├── models.py            # Neural network model definitions
├── strategy.py          # Flower client and strategy implementations
├── utils.py             # Utility functions for data and model handling
├── inference.py         # Model inference script
├── training.py          # Interactive training launcher
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── models/             # Saved models (created automatically)
└── data/               # Dataset cache (created automatically)
```

## Training Examples

### Quick Start Examples
```bash
# Basic CNN training on MNIST
python main.py --model cnn --dataset mnist --epochs 20 --gpu

# SNN with custom timesteps on CIFAR-10
python main.py --model snn --dataset cifar10 --snn_timesteps 30 --epochs 25 --gpu

# Non-IID training with FedProx
python main.py --model mlp --dataset mnist --strategy fedprox --fedprox_mu 0.1 --epochs 30
```

### Advanced Training
```bash
# Large-scale federated learning with learning rate scheduling
python main.py \
    --model cnn \
    --dataset cifar10 \
    --strategy fedadam \
    --num_users 20 \
    --epochs 50 \
    --use_lr_scheduler \
    --warmup_epochs 3 \
    --gpu
```

## Model Inference Features

- **Single Image Prediction**: Get top-k predictions with confidence scores
- **Batch Processing**: Process entire folders with progress tracking
- **Automatic Preprocessing**: Dataset-specific image transformations
- **Summary Statistics**: Accuracy metrics and class distribution analysis
- **Multiple Formats**: Support for PNG, JPG, JPEG, BMP, TIFF

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Flower 1.0+
- SNNTorch (for SNN models)
- See [requirements.txt](requirements.txt) for complete list

## Performance Tips

1. **GPU Usage**: Use `--gpu` flag for CUDA acceleration
2. **Memory Optimization**: Reduce `--local_bs` if encountering OOM errors
3. **Non-IID Data**: Use `--strategy fedprox` or `--auto_switch_fedprox`
4. **SNN Training**: Lower learning rates work better (auto-adjusted)
5. **Convergence**: Enable `--use_lr_scheduler` for better training dynamics

## Troubleshooting

- **CUDA Out of Memory**: Reduce batch size or number of clients
- **Strategy Import Errors**: Install with `pip install flwr[strategies]`
- **Poor Convergence**: Try different strategies or enable LR scheduling
- **Data Distribution Warnings**: Normal for non-IID scenarios

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Flower](https://flower.dev/) — for the federated learning infrastructure
- [SNNTorch](https://snntorch.readthedocs.io/) — for spiking neural network modeling
- [PyTorch](https://pytorch.org/) — for the deep learning backend
- [OpenAI ChatGPT](https://openai.com/chatgpt) — for assistance in code debugging, design.
- [Anthropic Claude](https://www.anthropic.com/index/claude) — for research support, code suggestions, and implementation guidance
- [Cursor AI](https://www.cursor.com/) — for enhanced code documentation and developer productivity


> Parts of this project were developed with assistance from AI tools (ChatGPT & Claude). All code was verified, tested, and curated manually.
