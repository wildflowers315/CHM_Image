"""Training callbacks for monitoring and control."""

import json
import os
import time
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from pathlib import Path


class EarlyStoppingCallback:
    """Early stopping callback to prevent overfitting."""
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.001,
                 monitor: str = 'val_loss'):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum improvement to consider as improvement
            monitor: Metric to monitor for improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value = None
        self.wait_count = 0
        self.stopped_epoch = 0
        
    def __call__(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of current metrics
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.monitor not in metrics:
            return False
        
        current_value = metrics[self.monitor]
        
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        # Check for improvement (assuming lower is better for loss)
        if self.monitor.endswith('_loss'):
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = current_value
            self.wait_count = 0
        else:
            self.wait_count += 1
        
        if self.wait_count >= self.patience:
            self.stopped_epoch = epoch
            return True
        
        return False


class TrainingLogger:
    """Logger for training progress and metrics."""
    
    def __init__(self, log_level: str = 'INFO'):
        """
        Initialize training logger.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.log_level = log_level
        self.logs: List[str] = []
        self.start_time = time.time()
        self.epoch_times: List[float] = []
        
    def log_info(self, message: str) -> None:
        """Log an info message."""
        timestamp = time.time() - self.start_time
        log_entry = f"[{timestamp:.2f}s] INFO: {message}"
        self.logs.append(log_entry)
        print(f"📝 {message}")
    
    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        timestamp = time.time() - self.start_time
        log_entry = f"[{timestamp:.2f}s] WARNING: {message}"
        self.logs.append(log_entry)
        print(f"⚠️ {message}")
    
    def log_error(self, message: str) -> None:
        """Log an error message."""
        timestamp = time.time() - self.start_time
        log_entry = f"[{timestamp:.2f}s] ERROR: {message}"
        self.logs.append(log_entry)
        print(f"❌ {message}")
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log training metrics for an epoch."""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.log_info(f"Epoch {epoch}: {metrics_str}")
    
    def log_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Log the start of an epoch."""
        self.epoch_start_time = time.time()
        self.log_info(f"Starting epoch {epoch}/{total_epochs}")
    
    def log_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log the end of an epoch."""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        self.log_metrics(epoch, metrics)
        self.log_info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
    
    def save_log(self, filepath: Path) -> None:
        """Save logs to file."""
        with open(filepath, 'w') as f:
            f.write("\\n".join(self.logs))
    
    def get_average_epoch_time(self) -> float:
        """Get average time per epoch."""
        if not self.epoch_times:
            return 0.0
        return sum(self.epoch_times) / len(self.epoch_times)
    
    def estimate_remaining_time(self, current_epoch: int, total_epochs: int) -> float:
        """Estimate remaining training time."""
        if not self.epoch_times:
            return 0.0
        avg_time = self.get_average_epoch_time()
        remaining_epochs = total_epochs - current_epoch
        return avg_time * remaining_epochs

class EarlyStoppingCallback2:
    """
    Patience-based early stopping with best model preservation.
    
    Features:
    - Configurable patience (default: 15 epochs)
    - Best validation loss tracking
    - Automatic model checkpoint saving
    - Learning rate scheduling integration
    """
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, 
                 restore_best_weights: bool = True, checkpoint_dir: str = None):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Restore best model weights on stop
            checkpoint_dir: Directory to save checkpoints
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.checkpoint_dir = checkpoint_dir
        
        self.best_loss = float('inf')
        self.best_weights = None
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            
    def __call__(self, epoch: int, val_loss: float, model: nn.Module, 
                 optimizer: torch.optim.Optimizer = None) -> bool:
        """
        Check early stopping criteria.
        
        Args:
            epoch: Current epoch number
            val_loss: Current validation loss
            model: Model to potentially save
            optimizer: Optimizer state to save
            
        Returns:
            True if training should stop, False otherwise
        """
        improved = val_loss < (self.best_loss - self.min_delta)
        
        if improved:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            
            # Save best weights
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # Save checkpoint
            if self.checkpoint_dir:
                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                self.save_checkpoint(model, optimizer, epoch, checkpoint_path, is_best=True)
                
            print(f"🎯 New best validation loss: {val_loss:.6f} (epoch {epoch})")
            
        else:
            self.epochs_without_improvement += 1
            
        # Check if we should stop
        should_stop = self.epochs_without_improvement >= self.patience
        
        if should_stop:
            print(f"⏹️  Early stopping triggered after {self.patience} epochs without improvement")
            print(f"📈 Best validation loss: {self.best_loss:.6f} (epoch {self.best_epoch})")
            
            # Restore best weights
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
                print("✅ Restored best model weights")
                
        return should_stop

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, checkpoint_path: str, is_best: bool = False):
        """Save comprehensive training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'epochs_without_improvement': self.epochs_without_improvement
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            # Also save as latest checkpoint
            latest_path = os.path.join(os.path.dirname(checkpoint_path), 'latest.pth')
            torch.save(checkpoint, latest_path)

class TrainingLogger2:
    """
    Comprehensive training metrics tracking and visualization.
    
    Features:
    - Loss curve tracking (train/validation)
    - Training time and resource monitoring
    - Automatic visualization generation
    - JSON metrics export
    """
    
    def __init__(self, output_dir: str, log_frequency: int = 10):
        """Initialize training logger."""
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_frequency = log_frequency
        self.epoch_logs = []
        self.batch_logs = []
        self.start_time = None
        
    def start_training(self):
        """Mark the start of training."""
        self.start_time = time.time()
        
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  learning_rate: float, epoch_time: float, gpu_memory: float = None):
        """Log epoch-level metrics."""
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate,
            'epoch_time': epoch_time,
            'total_time': time.time() - self.start_time if self.start_time else 0
        }
        
        if gpu_memory is not None:
            log_entry['gpu_memory_gb'] = gpu_memory
            
        self.epoch_logs.append(log_entry)
        
        # Print progress
        if epoch % self.log_frequency == 0:
            print(f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {learning_rate:.2e} | Time: {epoch_time:.1f}s")
    
    def log_batch(self, epoch: int, batch_idx: int, batch_loss: float, batch_size: int):
        """Log batch-level metrics."""
        log_entry = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'batch_loss': batch_loss,
            'batch_size': batch_size,
            'timestamp': time.time()
        }
        
        self.batch_logs.append(log_entry)
    
    def generate_loss_curves(self) -> str:
        """Generate and save loss curve visualizations."""
        if not self.epoch_logs:
            return None
            
        try:
            import matplotlib.pyplot as plt
            
            epochs = [log['epoch'] for log in self.epoch_logs]
            train_losses = [log['train_loss'] for log in self.epoch_logs]
            val_losses = [log['val_loss'] for log in self.epoch_logs]
            
            plt.figure(figsize=(12, 8))
            
            # Loss curves
            plt.subplot(2, 2, 1)
            plt.plot(epochs, train_losses, label='Training', alpha=0.8)
            plt.plot(epochs, val_losses, label='Validation', alpha=0.8)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Learning rate
            plt.subplot(2, 2, 2)
            learning_rates = [log['learning_rate'] for log in self.epoch_logs]
            plt.plot(epochs, learning_rates)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            
            # Training time
            plt.subplot(2, 2, 3)
            epoch_times = [log['epoch_time'] for log in self.epoch_logs]
            plt.plot(epochs, epoch_times)
            plt.xlabel('Epoch')
            plt.ylabel('Time (seconds)')
            plt.title('Epoch Training Time')
            plt.grid(True, alpha=0.3)
            
            # GPU memory (if available)
            plt.subplot(2, 2, 4)
            if 'gpu_memory_gb' in self.epoch_logs[0]:
                gpu_memory = [log['gpu_memory_gb'] for log in self.epoch_logs]
                plt.plot(epochs, gpu_memory)
                plt.xlabel('Epoch')
                plt.ylabel('GPU Memory (GB)')
                plt.title('GPU Memory Usage')
            else:
                # Total time curve
                total_times = [log['total_time'] for log in self.epoch_logs]
                plt.plot(epochs, total_times)
                plt.xlabel('Epoch')
                plt.ylabel('Total Time (seconds)')
                plt.title('Cumulative Training Time')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            curves_path = self.logs_dir / 'loss_curves.png'
            plt.savefig(curves_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(curves_path)
            
        except ImportError:
            print("Warning: matplotlib not available for loss curve generation")
            return None
    
    def export_metrics(self) -> str:
        """Export comprehensive training metrics to JSON."""
        metrics = {
            'training_summary': {
                'total_epochs': len(self.epoch_logs),
                'total_time': self.epoch_logs[-1]['total_time'] if self.epoch_logs else 0,
                'final_train_loss': self.epoch_logs[-1]['train_loss'] if self.epoch_logs else None,
                'final_val_loss': self.epoch_logs[-1]['val_loss'] if self.epoch_logs else None,
                'best_val_loss': min(log['val_loss'] for log in self.epoch_logs) if self.epoch_logs else None
            },
            'epoch_logs': self.epoch_logs,
            'batch_logs': self.batch_logs
        }
        
        # Save metrics
        metrics_path = self.logs_dir / 'training_log.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        return str(metrics_path)
