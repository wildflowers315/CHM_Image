"""Training callbacks for monitoring and control."""

import time
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