"""Training utilities and helper functions."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union


def load_training_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load training configuration from file.
    
    Args:
        config_path: Path to config file (JSON or YAML)
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return config


def save_training_results(results: Dict[str, Any], 
                         output_path: Union[str, Path]) -> None:
    """
    Save training results to JSON file.
    
    Args:
        results: Training results dictionary
        output_path: Path to save results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if hasattr(value, 'item'):  # numpy scalar
            serializable_results[key] = value.item()
        elif hasattr(value, 'tolist'):  # numpy array
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def create_output_directory(base_dir: str, model_type: str) -> Path:
    """
    Create standardized output directory structure.
    
    Args:
        base_dir: Base output directory
        model_type: Type of model (rf, mlp, 2d_unet, 3d_unet)
        
    Returns:
        Path to model-specific output directory
    """
    output_dir = Path(base_dir) / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'predictions').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    
    return output_dir


def validate_training_inputs(patch_files: list, 
                           model_type: str,
                           **kwargs) -> None:
    """
    Validate training inputs and parameters.
    
    Args:
        patch_files: List of patch file paths
        model_type: Type of model
        **kwargs: Additional parameters to validate
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate model type
    valid_models = ['rf', 'mlp', '2d_unet', '3d_unet']
    if model_type not in valid_models:
        raise ValueError(f"Invalid model type: {model_type}. Must be one of {valid_models}")
    
    # Validate patch files
    if not patch_files:
        raise ValueError("No patch files provided")
    
    for patch_file in patch_files:
        if not Path(patch_file).exists():
            raise ValueError(f"Patch file not found: {patch_file}")
    
    # Validate device for neural networks
    if model_type in ['2d_unet', '3d_unet']:
        device = kwargs.get('device', 'cpu')
        if device == 'cuda':
            try:
                import torch
                if not torch.cuda.is_available():
                    raise ValueError("CUDA requested but not available")
            except ImportError:
                raise ValueError("PyTorch required for U-Net models")


def get_model_info(model_type: str, **kwargs) -> Dict[str, Any]:
    """
    Get model-specific information and default parameters.
    
    Args:
        model_type: Type of model
        **kwargs: Additional parameters
        
    Returns:
        Model information dictionary
    """
    base_info = {
        'model_type': model_type,
        'framework': None,
        'supports_temporal': False,
        'requires_sparse_data': False,
        'default_params': {}
    }
    
    if model_type == 'rf':
        base_info.update({
            'framework': 'sklearn',
            'supports_temporal': True,
            'requires_sparse_data': True,
            'default_params': {
                'n_estimators': 100,
                'max_depth': None,
                'random_state': 42
            }
        })
    elif model_type == 'mlp':
        base_info.update({
            'framework': 'pytorch',
            'supports_temporal': True,
            'requires_sparse_data': True,
            'default_params': {
                'hidden_sizes': [512, 256, 128],
                'dropout_rate': 0.2,
                'learning_rate': 0.001
            }
        })
    elif model_type == '2d_unet':
        base_info.update({
            'framework': 'pytorch',
            'supports_temporal': False,
            'requires_sparse_data': False,
            'default_params': {
                'base_channels': 64,
                'learning_rate': 0.001,
                'batch_size': 8
            }
        })
    elif model_type == '3d_unet':
        base_info.update({
            'framework': 'pytorch',
            'supports_temporal': True,
            'requires_sparse_data': False,
            'default_params': {
                'base_channels': 32,
                'learning_rate': 0.0005,
                'batch_size': 4
            }
        })
    
    return base_info