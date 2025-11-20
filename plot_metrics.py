"""Simple script to plot training metrics from JSON logs."""
import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def plot_metrics(metrics_file: Path, output_dir: Path = None):
    """Plot training and validation metrics from JSON file."""
    with open(metrics_file) as f:
        history = json.load(f)
    
    epochs = [h["epoch"] for h in history]
    
    # Extract metrics
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["loss"] for h in history]
    val_auprc = [h["val"]["auprc"] for h in history]
    val_f1 = [h["val"]["f1"] for h in history]
    val_acc = [h["val"]["accuracy"] for h in history]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # Loss
    axes[0, 0].plot(epochs, train_loss, label='Train', marker='o')
    axes[0, 0].plot(epochs, val_loss, label='Val', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # AUPRC
    axes[0, 1].plot(epochs, val_auprc, label='Val AUPRC', marker='o', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUPRC')
    axes[0, 1].set_title('Validation AUPRC')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1
    axes[1, 0].plot(epochs, val_f1, label='Val F1', marker='o', color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 1].plot(epochs, val_acc, label='Val Accuracy', marker='o', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "training_metrics.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from JSON")
    parser.add_argument("metrics_file", type=Path, help="Path to metrics.json file")
    parser.add_argument("--output-dir", type=Path, help="Directory to save plot (if not set, displays plot)")
    args = parser.parse_args()
    
    if not args.metrics_file.exists():
        print(f"Error: {args.metrics_file} not found")
        return
    
    plot_metrics(args.metrics_file, args.output_dir)


if __name__ == "__main__":
    main()

