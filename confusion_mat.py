import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader
import os
from datetime import datetime
    
# Import your custom modules
from utils import load_data, load_model

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_trained_model(model_path):
    """Load the trained model from checkpoint using your utils function."""
    try:
        model, checkpoint = load_model(model_path)
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Model: {checkpoint['model_name'].upper()}")
        print(f"Dataset: {checkpoint['dataset'].upper()}")
        print(f"Clients: {checkpoint['args']['num_users']}")
        print(f"Rounds: {checkpoint['args']['epochs']}")
        print(f"Strategy: {checkpoint['args']['strategy'].upper()}")
        if checkpoint['model_name'] == 'snn':
            print(f"SNN Timesteps: {checkpoint['snn_timesteps']}")
        print(f"Distribution: {'IID' if checkpoint['args']['iid'] else 'Non-IID'}")
        
        return model, checkpoint
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def get_test_loader(dataset_name, batch_size=1000):
    """Get test data loader using your data loading function."""
    try:
        _, test_data = load_data(dataset_name)
        test_loader = DataLoader(
            test_data, 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=True if device.type == 'cuda' else False
        )
        return test_loader
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None

def get_predictions(model, test_loader, model_name):
    """Generate predictions using the loaded model."""
    all_predictions = []
    all_labels = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for _, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            if model_name == 'snn':
                # For SNN models, get membrane potentials and sum over time steps
                _, mem_rec = model(data)
                if isinstance(mem_rec, list):
                    mem_rec = torch.stack(mem_rec, dim=0)
                outputs = mem_rec.sum(dim=0)
            else:
                # For CNN/MLP models
                outputs = model(data)
            
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
                
    return np.array(all_predictions), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, dataset_name, model_info, save_path=None):
    """Create and plot confusion matrix with dataset-specific class names."""
    if dataset_name == 'mnist':
        classes = [str(i) for i in range(10)]
        title_suffix = "MNIST Digits"
    elif dataset_name == 'cifar10':
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
        title_suffix = "CIFAR-10 Objects"
    else:
        classes = [str(i) for i in range(10)]
        title_suffix = "Classes"
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    
    # Create detailed title
    model_name = model_info['model_name'].upper()
    strategy = model_info['args']['strategy'].upper()
    distribution = 'IID' if model_info['args']['iid'] else 'Non-IID'
    clients = model_info['args']['num_users']
    rounds = model_info['args']['epochs']
    
    title = f'Confusion Matrix - {model_name} {title_suffix}\n'
    title += f'Strategy: {strategy} | Distribution: {distribution} | '
    title += f'Clients: {clients} | Rounds: {rounds}'
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Rotate x-axis labels if needed
    if dataset_name == 'cifar10':
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return cm

def save_detailed_results(cm, accuracy, y_true, y_pred, dataset_name, model_info, results_dir="results"):
    """Save detailed analysis results to text file."""
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename
    model_name = model_info['model_name']
    strategy = model_info['args']['strategy']
    clients = model_info['args']['num_users']
    rounds = model_info['args']['epochs']
    distribution = 'noniid' if not model_info['args']['iid'] else 'iid'
    
    filename = f"results_{model_name}_{dataset_name}_{strategy}_clients{clients}_rounds{rounds}_{distribution}.txt"
    filepath = os.path.join(results_dir, filename)
    
    # Class names
    if dataset_name == 'mnist':
        class_names = [f'Digit {i}' for i in range(10)]
    elif dataset_name == 'cifar10':
        class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
                      'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    else:
        class_names = [f'Class {i}' for i in range(10)]
    
    with open(filepath, 'w') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("CONFUSION MATRIX ANALYSIS RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model Information
        f.write("MODEL CONFIGURATION\n")
        f.write("-"*40 + "\n")
        f.write(f"Model Type: {model_info['model_name'].upper()}\n")
        f.write(f"Dataset: {dataset_name.upper()}\n")
        f.write(f"Strategy: {strategy.upper()}\n")
        f.write(f"Number of Clients: {clients}\n")
        f.write(f"Number of Rounds: {rounds}\n")
        f.write(f"Data Distribution: {'Non-IID' if not model_info['args']['iid'] else 'IID'}\n")
        f.write(f"Learning Rate: {model_info['args']['lr']}\n")
        f.write(f"Local Epochs: {model_info['args']['local_ep']}\n")
        f.write(f"Local Batch Size: {model_info['args']['local_bs']}\n")
        if model_info['model_name'] == 'snn':
            f.write(f"SNN Timesteps: {model_info['snn_timesteps']}\n")
        if strategy == 'fedprox':
            f.write(f"FedProx Mu: {model_info['args'].get('fedprox_mu', 'N/A')}\n")
        f.write("\n")
        
        # Overall Performance
        f.write("OVERALL PERFORMANCE\n")
        f.write("-"*40 + "\n")
        f.write(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Total Test Samples: {len(y_true)}\n")
        f.write(f"Correctly Classified: {np.sum(y_true == y_pred)}\n")
        f.write(f"Misclassified: {np.sum(y_true != y_pred)}\n\n")
        
        # Per-Class Performance
        f.write("PER-CLASS PERFORMANCE\n")
        f.write("-"*40 + "\n")
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        class_support = cm.sum(axis=1)
        
        f.write(f"{'Class':<15} {'Accuracy':<10} {'Support':<8} {'Correct':<8}\n")
        f.write("-" * 45 + "\n")
        
        for i, (acc, support, correct) in enumerate(zip(class_accuracy, class_support, cm.diagonal())):
            f.write(f"{class_names[i]:<15} {acc:.4f}     {support:<8} {correct:<8}\n")
        
        f.write(f"\nBest Class: {class_names[np.argmax(class_accuracy)]} ({np.max(class_accuracy):.4f})\n")
        f.write(f"Worst Class: {class_names[np.argmin(class_accuracy)]} ({np.min(class_accuracy):.4f})\n\n")
        
        # Confusion Matrix
        f.write("CONFUSION MATRIX\n")
        f.write("-"*40 + "\n")
        f.write("Rows: True Labels, Columns: Predicted Labels\n\n")
        
        # Header
        f.write("     ")
        for i in range(10):
            f.write(f"{i:>6}")
        f.write("\n")
        
        # Matrix rows
        for i in range(10):
            f.write(f"{i:>3}: ")
            for j in range(10):
                f.write(f"{cm[i,j]:>6}")
            f.write("\n")
        f.write("\n")
        
        # Classification Report
        from sklearn.metrics import classification_report
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("-"*40 + "\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    print(f"Detailed results saved to {filepath}")
    return filepath

def calculate_detailed_metrics(y_true, y_pred, dataset_name):
    """Calculate and display comprehensive metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Class names for better readability
    if dataset_name == 'mnist':
        target_names = [f'Digit {i}' for i in range(10)]
    elif dataset_name == 'cifar10':
        target_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
                       'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    else:
        target_names = [f'Class {i}' for i in range(10)]
    
    print(f"\n{'-'*60}")
    print("CLASSIFICATION REPORT")
    print(f"{'-'*60}")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    
    return accuracy

def analyze_per_class_performance(cm, dataset_name):
    """Analyze per-class performance from confusion matrix."""
    if dataset_name == 'cifar10':
        class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
                      'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    else:
        class_names = [f'Class {i}' for i in range(10)]
    
    print(f"\n{'-'*60}")
    print("PER-CLASS ACCURACY")
    print(f"{'-'*60}")
    
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    class_support = cm.sum(axis=1)
    
    # Sort by accuracy for better visualization
    sorted_indices = np.argsort(class_accuracy)
    
    print(f"{'Class':<12} {'Accuracy':<10} {'Support':<8} {'Correct':<8}")
    print("-" * 40)
    
    for idx in sorted_indices:
        acc = class_accuracy[idx]
        support = class_support[idx]
        correct = cm[idx, idx]
        
        print(f"{class_names[idx]:<12} {acc:.4f}     {support:<8} {correct:<8}")
    
    print(f"\nBest performing class: {class_names[sorted_indices[-1]]} ({class_accuracy[sorted_indices[-1]]:.4f})")
    print(f"Worst performing class: {class_names[sorted_indices[0]]} ({class_accuracy[sorted_indices[0]]:.4f})")

def scan_models_directory(models_dir="models"):
    """Scan the models directory and return available model files."""

    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' not found.")
        return []
    
    model_files = []
    for filename in os.listdir(models_dir):
        if filename.endswith('.pth'):
            model_files.append(filename)
    
    return sorted(model_files)

def select_model(models_dir="models"):
    """Let user select a model from available models."""
    model_files = scan_models_directory(models_dir)
    
    if not model_files:
        print(f"No .pth model files found in '{models_dir}' directory.")
        return None
    
    print(f"\nFound {len(model_files)} model(s) in '{models_dir}' directory:")
    print("-" * 80)
    
    for i, filename in enumerate(model_files):
        # Extract info from filename for display
        parts = filename.replace('.pth', '').split('_')
        if len(parts) >= 4:
            model_type = parts[0].upper()
            dataset = parts[1].upper()
            strategy = parts[2].upper()
            
            # Extract additional info
            info_parts = parts[3:]
            clients = rounds = distribution = "Unknown"
            
            for part in info_parts:
                if part.startswith('clients'):
                    clients = part.replace('clients', '')
                elif part.startswith('rounds'):
                    rounds = part.replace('rounds', '')
                elif part == 'noniid':
                    distribution = "Non-IID"
            
            if distribution == "Unknown":
                distribution = "IID"
            
            print(f"[{i+1}] {filename}")
            print(f"    Model: {model_type} | Dataset: {dataset} | Strategy: {strategy}")
            print(f"    Clients: {clients} | Rounds: {rounds} | Distribution: {distribution}")
        else:
            print(f"[{i+1}] {filename}")
        print()
    
    while True:
        try:
            choice = input(f"Select a model (1-{len(model_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(model_files):
                selected_file = model_files[choice_idx]
                model_path = os.path.join(models_dir, selected_file)
                print(f"\nSelected: {selected_file}")
                return model_path
            else:
                print(f"Please enter a number between 1 and {len(model_files)}")
                
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None

def main():
    """Main execution function."""
    print("="*60)
    print("CONFUSION MATRIX GENERATION")
    print("="*60)
    
    # Let user select model
    model_path = select_model("models")
    if model_path is None:
        print("No model selected. Exiting.")
        return
    
    # Load model
    print(f"\nLoading selected model: {model_path}")
    model, model_info = load_trained_model(model_path)
    if model is None:
        print("Failed to load model. Please check the model path.")
        return
    
    # Extract model information
    model_name = model_info['model_name']
    dataset_name = model_info['dataset']
    
    # Load test data
    print(f"\nLoading {dataset_name.upper()} test data...")
    test_loader = get_test_loader(dataset_name)
    if test_loader is None:
        print("Failed to load test data.")
        return
    
    print(f"Test dataset size: {len(test_loader.dataset)} samples")
    
    # Generate predictions
    predictions, labels = get_predictions(model, test_loader, model_name)
    
    # Create confusion matrix
    print("\nCreating confusion matrix...")
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", f"confusion_matrix_{model_info['model_name']}_{dataset_name}_{model_info['args']['strategy']}.png")
    cm = plot_confusion_matrix(
        labels, predictions, dataset_name, model_info, save_path
    )

    
    # Calculate comprehensive metrics
    accuracy = calculate_detailed_metrics(labels, predictions, dataset_name)
    
    # Analyze per-class performance
    analyze_per_class_performance(cm, dataset_name)
    
    # Save detailed results to text file
    save_detailed_results(cm, accuracy, labels, predictions, dataset_name, model_info, "results")
    
    # Print raw confusion matrix
    print(f"\n{'-'*60}")
    print("RAW CONFUSION MATRIX")
    print(f"{'-'*60}")
    print(cm)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"Final Accuracy: {accuracy*100:.2f}%")
    print(f"Results saved to 'results/' directory")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()