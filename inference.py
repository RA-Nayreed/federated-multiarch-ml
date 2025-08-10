"""
Inference script for saved federated learning models
Enhanced with proper error handling and robustness
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
import os
import sys

try:
    from utils import load_model
except ImportError as e:
    print(f"Error importing utilities: {e}")
    print("Make sure utils.py is available in the project directory or your PYTHONPATH")
    sys.exit(1)

def preprocess_image(image_path, dataset):
    """Preprocess image for inference with error handling"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
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
        
        return transform(image).unsqueeze(0)
    
    except Exception as e:
        raise ValueError(f"Error preprocessing image {image_path}: {e}")

def get_device_safe(model):
    """Safely get model device with fallback"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        print("Warning: Model has no parameters, using CPU")
        return torch.device('cpu')

def predict(model, image_tensor, model_name, dataset):
    """Make prediction on image with improved error handling"""
    try:
        device = get_device_safe(model)
        image_tensor = image_tensor.to(device)
        
        model.eval()  # Ensure model is in evaluation mode
        
        with torch.no_grad():
            if model_name == 'snn':
                # Handle SNN models that return spikes and membrane potentials
                outputs = model(image_tensor)
                if isinstance(outputs, tuple):
                    spk_rec, mem_rec = outputs
                    outputs = mem_rec[-1]  
                else:
                    # If it's not a tuple, assume it's the output directly
                    pass
            else:
                outputs = model(image_tensor)
        
        # Ensure outputs is 2D (batch_size, num_classes)
        if outputs.dim() == 3:
            outputs = outputs.view(outputs.size(0), -1)
        
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()
    
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")

def batch_inference(model, image_folder, model_name, dataset):
    """Run inference on all images in a folder with progress tracking"""
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")
    
    results = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"No image files found in {image_folder}")
        return results
    
    print(f"Processing {len(image_files)} images...")
    
    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, filename)
        try:
            image_tensor = preprocess_image(image_path, dataset)
            predicted_class, confidence, _ = predict(model, image_tensor, model_name, dataset)
            results.append({
                'filename': filename,
                'predicted_class': predicted_class,
                'confidence': confidence
            })
            
            # Progress indicator
            if i % 10 == 0 or i == len(image_files):
                print(f"Processed {i}/{len(image_files)} images")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return results

def validate_args(args):
    """Validate command line arguments"""
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    if args.image_path and not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    
    if args.image_folder and not os.path.exists(args.image_folder):
        raise FileNotFoundError(f"Image folder not found: {args.image_folder}")
    
    if not args.image_path and not args.image_folder:
        raise ValueError("Please provide either --image_path or --image_folder")
    
    if args.image_path and args.image_folder:
        raise ValueError("Please provide either --image_path OR --image_folder, not both")

def get_class_names(dataset):
    """Get class names for the dataset"""
    class_names_dict = {
        'mnist': [str(i) for i in range(10)],
        'cifar10': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    }
    
    return class_names_dict.get(dataset, [f'Class_{i}' for i in range(10)])

def main():
    parser = argparse.ArgumentParser(description='Inference with saved federated model')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to saved model (.pth file)')
    parser.add_argument('--image_path', type=str, 
                       help='Path to input image')
    parser.add_argument('--image_folder', type=str, 
                       help='Path to folder of images')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], 
                       help='Dataset type (if not specified, will try to get from checkpoint)')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top predictions to show (default: 3)')

    args = parser.parse_args()

    try:

        validate_args(args)
        
        print(f"Loading model from: {args.model_path}")
        
        # Load model
        model, checkpoint = load_model(args.model_path)
        
        # Get dataset and model info
        dataset = args.dataset or checkpoint.get('dataset')
        model_name = checkpoint.get('model_name', 'unknown')
        
        if not dataset:
            raise ValueError("Dataset not specified and not found in checkpoint. Please use --dataset")
        
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset}")
        print(f"Device: {get_device_safe(model)}")
        
        # Get class names
        class_names = get_class_names(dataset)

        if args.image_path:
            # Single image inference
            print(f"\nProcessing single image: {args.image_path}")
            
            image_tensor = preprocess_image(args.image_path, dataset)
            predicted_class, confidence, probabilities = predict(model, image_tensor, model_name, dataset)

            print("\n" + '='*50)
            print("PREDICTION RESULTS")
            print('='*50)
            print(f"Image: {os.path.basename(args.image_path)}")
            print(f"Predicted class: {predicted_class} ({class_names[predicted_class]})")
            print(f"Confidence: {confidence:.4f}")
            
            print(f"\nTop {args.top_k} predictions:")
            top_k_indices = np.argsort(probabilities)[-args.top_k:][::-1]
            for i, idx in enumerate(top_k_indices):
                class_name = class_names[idx] if idx < len(class_names) else f'Class_{idx}'
                print(f"{i+1:2d}. {class_name:<12}: {probabilities[idx]:.4f}")

        elif args.image_folder:
            # Folder-based inference
            print(f"\nProcessing images in folder: {args.image_folder}")
            
            results = batch_inference(model, args.image_folder, model_name, dataset)
            
            if results:
                print("\n" + '='*70)
                print(f"BATCH INFERENCE RESULTS ({len(results)} images)")
                print('='*70)
                
                # Sort by confidence (highest first)
                results.sort(key=lambda x: x['confidence'], reverse=True)
                
                for r in results:
                    class_name = class_names[r['predicted_class']] if r['predicted_class'] < len(class_names) else f"Class_{r['predicted_class']}"
                    print(f"{r['filename']:<25} -> {class_name:<12} (Conf: {r['confidence']:.4f})")
                
                # Summary statistics
                confidences = [r['confidence'] for r in results]
                print("\n" + '='*70)
                print("SUMMARY STATISTICS")
                print('='*70)
                print(f"Total images processed: {len(results)}")
                print(f"Average confidence: {np.mean(confidences):.4f}")
                print(f"Min confidence: {np.min(confidences):.4f}")
                print(f"Max confidence: {np.max(confidences):.4f}")
                
                # Class distribution
                classes = [r['predicted_class'] for r in results]
                unique_classes, counts = np.unique(classes, return_counts=True)
                print("\nClass distribution:")
                for class_id, count in zip(unique_classes, counts):
                    class_name = class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}'
                    print(f"  {class_name:<12}: {count:3d} images ({count/len(results)*100:.1f}%)")
            else:
                print("No images were successfully processed.")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
    
