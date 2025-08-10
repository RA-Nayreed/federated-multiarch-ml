"""
Inference script for saved federated learning models.

This module provides robust inference capabilities for trained federated learning models,
supporting both single image and batch processing with comprehensive error handling
and validation.

Features:
    - Single image inference with detailed predictions
    - Batch processing of image folders
    - Support for MNIST and CIFAR-10 datasets
    - Automatic model device detection and optimization
    - Comprehensive error handling and validation
    - Progress tracking for batch operations
    - Detailed statistics and confidence reporting

Supported Models:
    - MLP: Multi-Layer Perceptron
    - CNN: Convolutional Neural Network  
    - SNN: Spiking Neural Network

Usage:
    python inference.py --model_path models/cnn_mnist_fedavg.pth --image_path test.png
    python inference.py --model_path models/snn_cifar10_fedprox.pth --image_folder test_images/
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
import sys
from typing import Tuple, List, Dict, Any

from pathlib import Path

try:
    from utils import load_model
except ImportError as e:
    print(f"Error importing utilities: {e}")
    print("Make sure utils.py is available in the project directory or your PYTHONPATH")
    sys.exit(1)


def preprocess_image(image_path: str, dataset: str) -> torch.Tensor:
    """
    Preprocess image for inference with comprehensive error handling.
    
    """
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not image_path.is_file():
            raise ValueError(f"Path is not a file: {image_path}")
        
        image = Image.open(image_path)
        
        if dataset == 'mnist':
            if image.mode != 'L':
                image = image.convert('L')
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif dataset == 'cifar10':
            if image.mode != 'RGB':
                image = image.convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        processed_tensor = transform(image).unsqueeze(0)
        
        if torch.isnan(processed_tensor).any():
            raise ValueError("Preprocessed image contains NaN values")
        
        return processed_tensor
    
    except (OSError, IOError) as e:
        raise RuntimeError(f"Error loading image {image_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error preprocessing image {image_path}: {e}")


def get_device_safe(model: torch.nn.Module) -> torch.device:
    """
    Safely determine the device of a model with fallback to CPU.

    """
    try:
        if hasattr(model, 'device'):
            return model.device
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        print("Warning: Model has no parameters or device attribute, using CPU")
        return torch.device('cpu')


def predict(model: torch.nn.Module, image_tensor: torch.Tensor, 
           model_name: str, dataset: str) -> Tuple[int, float, np.ndarray]:
    """
    Make prediction on image with comprehensive error handling.
    
    """
    try:
        device = get_device_safe(model)
        image_tensor = image_tensor.to(device)
        
        if torch.isnan(image_tensor).any():
            raise ValueError("Input tensor contains NaN values")
        
        model.eval()
        
        with torch.no_grad():
            if model_name == 'snn':
                outputs = model(image_tensor)
                if isinstance(outputs, tuple):
                    spk_rec, mem_rec = outputs
                    if isinstance(mem_rec, torch.Tensor):
                        outputs = mem_rec[-1] if mem_rec.dim() > 2 else mem_rec
                    else:
                        raise ValueError("Unexpected SNN output format")
                else:
                    outputs = outputs
            else:
                outputs = model(image_tensor)
        
        if torch.isnan(outputs).any():
            raise ValueError("Model output contains NaN values")
        
        if outputs.dim() == 3:
            outputs = outputs.view(outputs.size(0), -1)
        
        if outputs.dim() != 2:
            raise ValueError(f"Unexpected output dimensions: {outputs.shape}")
        
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        if not (0 <= predicted_class < probabilities.size(1)):
            raise ValueError(f"Predicted class {predicted_class} out of range [0, {probabilities.size(1)})")
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()
    
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")


def batch_inference(model: torch.nn.Module, image_folder: str, model_name: str, 
                   dataset: str) -> List[Dict[str, Any]]:
    """
    Run inference on all images in a folder with progress tracking and error handling.
    
    """
    image_folder = Path(image_folder)
    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")
    
    if not image_folder.is_dir():
        raise ValueError(f"Path is not a directory: {image_folder}")
    
    results = []
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    image_files = [f for f in image_folder.iterdir() 
                   if f.is_file() and f.suffix.lower() in valid_extensions]
    
    if not image_files:
        raise ValueError(f"No valid image files found in {image_folder}")
    
    print(f"Processing {len(image_files)} images...")
    
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(image_files, 1):
        try:
            image_tensor = preprocess_image(str(image_path), dataset)
            predicted_class, confidence, _ = predict(model, image_tensor, model_name, dataset)
            
            results.append({
                'filename': image_path.name,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'filepath': str(image_path)
            })
            successful += 1
            
            if i % 10 == 0 or i == len(image_files):
                print(f"Processed {i}/{len(image_files)} images (Success: {successful}, Failed: {failed})")
                
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            failed += 1
            continue
    
    print(f"\nBatch processing completed: {successful} successful, {failed} failed")
    return results


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments with comprehensive checks.
    
    """
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    if args.image_path and not Path(args.image_path).exists():
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    
    if args.image_folder and not Path(args.image_folder).exists():
        raise FileNotFoundError(f"Image folder not found: {args.image_folder}")
    
    if not args.image_path and not args.image_folder:
        raise ValueError("Please provide either --image_path or --image_folder")
    
    if args.image_path and args.image_folder:
        raise ValueError("Please provide either --image_path OR --image_folder, not both")
    
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")
    
    if args.dataset and args.dataset not in ['mnist', 'cifar10']:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


def get_class_names(dataset: str) -> List[str]:
    """
    Get class names for the specified dataset.
    
    """
    class_names_dict = {
        'mnist': [str(i) for i in range(10)],
        'cifar10': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    }
    
    return class_names_dict.get(dataset, [f'Class_{i}' for i in range(10)])


def print_single_image_results(image_path: str, predicted_class: int, confidence: float, 
                              probabilities: np.ndarray, class_names: List[str], top_k: int) -> None:
    """
    Print formatted results for single image inference.
    
    Args:
        image_path: Path to the input image
        predicted_class: Predicted class index
        confidence: Prediction confidence
        probabilities: Class probabilities array
        class_names: List of class names
        top_k: Number of top predictions to show
    """
    print("\n" + '='*50)
    print("PREDICTION RESULTS")
    print('='*50)
    print(f"Image: {Path(image_path).name}")
    print(f"Predicted class: {predicted_class} ({class_names[predicted_class]})")
    print(f"Confidence: {confidence:.4f}")
    
    print(f"\nTop {top_k} predictions:")
    top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
    for i, idx in enumerate(top_k_indices):
        class_name = class_names[idx] if idx < len(class_names) else f'Class_{idx}'
        print(f"{i+1:2d}. {class_name:<12}: {probabilities[idx]:.4f}")


def print_batch_results(results: List[Dict[str, Any]], class_names: List[str]) -> None:
    """
    Print formatted results for batch inference with statistics.

    """
    if not results:
        print("No images were successfully processed.")
        return
    
    print("\n" + '='*70)
    print(f"BATCH INFERENCE RESULTS ({len(results)} images)")
    print('='*70)
    
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    for r in results:
        class_name = class_names[r['predicted_class']] if r['predicted_class'] < len(class_names) else f"Class_{r['predicted_class']}"
        print(f"{r['filename']:<25} -> {class_name:<12} (Conf: {r['confidence']:.4f})")
    
    confidences = [r['confidence'] for r in results]
    print("\n" + '='*70)
    print("SUMMARY STATISTICS")
    print('='*70)
    print(f"Total images processed: {len(results)}")
    print(f"Average confidence: {np.mean(confidences):.4f}")
    print(f"Min confidence: {np.min(confidences):.4f}")
    print(f"Max confidence: {np.max(confidences):.4f}")
    print(f"Confidence std: {np.std(confidences):.4f}")
    
    classes = [r['predicted_class'] for r in results]
    unique_classes, counts = np.unique(classes, return_counts=True)
    print("\nClass distribution:")
    for class_id, count in zip(unique_classes, counts):
        class_name = class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}'
        percentage = count / len(results) * 100
        print(f"  {class_name:<12}: {count:3d} images ({percentage:.1f}%)")


def main():
    """
    Main entry point for the inference script.
    
    This function handles command line argument parsing, model loading,
    and orchestrates the inference process for single images or batches.
    """
    parser = argparse.ArgumentParser(
        description='Inference with saved federated learning models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            %(prog)s --model_path models/cnn_mnist_fedavg.pth --image_path test.png
            %(prog)s --model_path models/snn_cifar10_fedprox.pth --image_folder test_images/
            %(prog)s --model_path models/mlp_mnist_feddyn.pth --image_path digit.png --top_k 5
        """
    )
    
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to saved model (.pth file)')
    parser.add_argument('--image_path', type=str, 
                       help='Path to input image for single inference')
    parser.add_argument('--image_folder', type=str, 
                       help='Path to folder of images for batch inference')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], 
                       help='Dataset type (auto-detected from checkpoint if not specified)')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top predictions to show (default: 3)')

    args = parser.parse_args()

    try:
        validate_args(args)
        
        print(f"Loading model from: {args.model_path}")
        
        model, checkpoint = load_model(args.model_path)
        
        dataset = args.dataset or checkpoint.get('dataset')
        model_name = checkpoint.get('model_name', 'unknown')
        
        if not dataset:
            raise ValueError("Dataset not specified and not found in checkpoint. Please use --dataset")
        
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset}")
        print(f"Device: {get_device_safe(model)}")
        
        class_names = get_class_names(dataset)

        if args.image_path:
            print(f"\nProcessing single image: {args.image_path}")
            
            image_tensor = preprocess_image(args.image_path, dataset)
            predicted_class, confidence, probabilities = predict(model, image_tensor, model_name, dataset)
            
            print_single_image_results(args.image_path, predicted_class, confidence, 
                                     probabilities, class_names, args.top_k)

        elif args.image_folder:
            print(f"\nProcessing images in folder: {args.image_folder}")
            
            results = batch_inference(model, args.image_folder, model_name, dataset)
            print_batch_results(results, class_names)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
    
