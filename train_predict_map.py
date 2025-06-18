import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
try:
    import torch
    import torch.optim as optim
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. 3D U-Net will not be functional.")

from dl_models import MLPRegressionModel, create_normalized_dataloader
import rasterio
from rasterio.mask import geometry_mask
from shapely.geometry import Point, box
from shapely.ops import transform
import geopandas as gpd
import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import warnings
import argparse
from tqdm import tqdm
import glob
warnings.filterwarnings('ignore')

from evaluate_predictions import calculate_metrics
try:
    # Try importing 3D U-Net directly
    exec(open('models/3d_unet.py').read())
except (ImportError, FileNotFoundError):
    print("Warning: 3D U-Net model not found. Creating placeholder functions.")
    def Height3DUNet(*args, **kwargs):
        raise ImportError("3D U-Net not available")
    def create_3d_unet(*args, **kwargs):
        raise ImportError("3D U-Net not available")
from data.normalization import (
    normalize_sentinel1, normalize_sentinel2, normalize_srtm_elevation,
    normalize_srtm_slope, normalize_srtm_aspect, normalize_alos2_dn,
    normalize_canopy_height, normalize_ndvi
)

def modified_huber_loss(pred: torch.Tensor, target: torch.Tensor, 
                       mask: Optional[torch.Tensor] = None, 
                       delta: float = 1.0, shift_radius: int = 1) -> torch.Tensor:
    """
    Modified Huber loss for 3D patches with shift awareness for sparse GEDI data.
    
    Args:
        pred: Predicted values [batch, height, width]
        target: Target values [batch, height, width] (sparse GEDI)
        mask: Valid pixel mask [batch, height, width]
        delta: Huber loss threshold
        shift_radius: Radius for spatial shift compensation
        
    Returns:
        Loss value
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for modified Huber loss")
    
    def huber_loss(x, y, delta=1.0):
        diff = x - y
        abs_diff = diff.abs()
        quadratic = torch.min(abs_diff, torch.tensor(delta, device=x.device))
        linear = abs_diff - quadratic
        return 0.5 * quadratic.pow(2) + delta * linear
    
    def generate_shifts(radius):
        """Generate all possible shifts within given radius"""
        shifts = [(0, 0)]  # Always include no shift
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                # Only include shifts within radius (using Euclidean distance)
                if (dx*dx + dy*dy) <= radius*radius:
                    shifts.append((dx, dy))
        return shifts
    
    # Generate shifts based on radius
    shifts = generate_shifts(shift_radius)
    
    best_loss = float('inf')
    for dx, dy in shifts:
        # Shift target
        shifted_target = torch.roll(target, shifts=(dx, dy), dims=(1, 2))
        
        # Compute loss only on valid GEDI pixels
        if mask is not None:
            shifted_mask = torch.roll(mask, shifts=(dx, dy), dims=(1, 2))
            valid_pixels = shifted_mask & (shifted_target > 0)  # GEDI pixels
        else:
            valid_pixels = shifted_target > 0
        
        if valid_pixels.sum() > 0:
            loss = huber_loss(
                pred[valid_pixels], 
                shifted_target[valid_pixels], 
                delta
            ).mean()
            best_loss = min(best_loss, loss.item())
    
    return torch.tensor(best_loss, requires_grad=True)

def load_patch_data(patch_path: str, normalize_bands: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Load 3D patch data from GeoTIFF file with normalization.
    
    Args:
        patch_path: Path to patch GeoTIFF file
        normalize_bands: Whether to apply band-specific normalization
        
    Returns:
        features: Feature array [bands, height, width]
        gedi_target: GEDI target array [height, width]
        band_info: Dictionary mapping band names to indices
    """
    with rasterio.open(patch_path) as src:
        data = src.read()  # [bands, height, width]
        band_descriptions = src.descriptions
        
        # Create band mapping
        band_info = {desc: i for i, desc in enumerate(band_descriptions) if desc}
        
        # Find GEDI band
        gedi_idx = None
        for i, desc in enumerate(band_descriptions):
            if desc and 'rh' in desc.lower():
                gedi_idx = i
                break
        
        if gedi_idx is None:
            raise ValueError("No GEDI (rh) band found in patch data")
        
        # Extract GEDI target
        gedi_target = data[gedi_idx].astype(np.float32)
        
        # Extract features (all bands except GEDI and forest mask)
        feature_indices = []
        for i, desc in enumerate(band_descriptions):
            if desc and desc not in ['rh', 'forest_mask']:
                feature_indices.append(i)
        
        features = data[feature_indices].astype(np.float32)
        
        # Apply normalization if requested
        if normalize_bands:
            for i, idx in enumerate(feature_indices):
                desc = band_descriptions[idx]
                if desc.startswith('S1_'):
                    features[i] = normalize_sentinel1(features[i])
                elif desc.startswith('B') and len(desc) <= 3:  # S2 bands
                    features[i] = normalize_sentinel2(features[i])
                elif 'elevation' in desc.lower():
                    features[i] = normalize_srtm_elevation(features[i])
                elif 'slope' in desc.lower():
                    features[i] = normalize_srtm_slope(features[i])
                elif 'aspect' in desc.lower():
                    features[i] = normalize_srtm_aspect(features[i])
                elif desc.startswith('ALOS2_'):
                    features[i] = normalize_alos2_dn(features[i])
                elif desc == 'NDVI':
                    features[i] = normalize_ndvi(features[i])
                elif desc.startswith('ch_'):
                    features[i] = normalize_canopy_height(features[i])
    
    return features, gedi_target, band_info

def load_patches_from_directory(patches_dir: str, pattern: str = "*.tif") -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Load all patch files from directory.
    
    Args:
        patches_dir: Directory containing patch files
        pattern: File pattern to match
        
    Returns:
        List of (features, gedi_target, patch_name) tuples
    """
    patch_files = glob.glob(os.path.join(patches_dir, pattern))
    patch_data = []
    
    for patch_file in tqdm(patch_files, desc="Loading patches"):
        try:
            features, gedi_target, _ = load_patch_data(patch_file)
            patch_name = os.path.basename(patch_file)
            patch_data.append((features, gedi_target, patch_name))
        except Exception as e:
            print(f"Error loading patch {patch_file}: {e}")
            continue
    
    return patch_data

def load_training_data(csv_path: str, mask_path: Optional[str] = None,
                      feature_names: Optional[list] = None, ch_col: str = 'rh') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from CSV file and optionally mask with forest mask.
    
    Args:
        csv_path: Path to training data CSV
        mask_path: Optional path to forest mask TIF
        
    Returns:
        X: Feature matrix
        y: Target variable (rh)
    """
    # Read training data
    df = pd.read_csv(csv_path)
    
    # Create GeoDataFrame from points
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df['longitude'], df['latitude'])],
        crs="EPSG:4326"
    )
    
    if mask_path:
        with rasterio.open(mask_path) as mask_src:
            # Check CRS
            mask_crs = mask_src.crs
            if mask_crs != gdf.crs:
                gdf = gdf.to_crs(mask_crs)
            
            # Get bounds of mask
            mask_bounds = box(*mask_src.bounds)
            
            # First filter points by mask bounds
            gdf_masked = gdf[gdf.geometry.within(mask_bounds)]
            
            if len(gdf_masked) == 0:
                raise ValueError("No training points fall within the mask bounds")
            else:
                gdf = gdf_masked
            
            # Convert points to pixel coordinates
            pts_pixels = []
            valid_indices = []
            for idx, point in enumerate(gdf.geometry):
                row, col = rasterio.transform.rowcol(mask_src.transform, 
                                                   point.x, 
                                                   point.y)
                if (0 <= row < mask_src.height and 
                    0 <= col < mask_src.width):
                    pts_pixels.append((row, col))
                    valid_indices.append(idx)
            
            if not pts_pixels:
                raise ValueError("No training points could be mapped to valid pixels")
            
            # Read forest mask values at pixel locations
            mask_values = [mask_src.read(1)[r, c] for r, c in pts_pixels]
            
            # Filter points by mask values
            mask_indices = [i for i, v in enumerate(mask_values) if v == 1]
            if not mask_indices:
                raise ValueError("No training points fall within the forest mask")
            
            final_indices = [valid_indices[i] for i in mask_indices]
            gdf = gdf.iloc[final_indices]
    
    # Convert back to original CRS if needed
    if mask_path and mask_crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    # Separate features and target
    df = pd.DataFrame(gdf.drop(columns='geometry'))
    y = df[ch_col].values
    
    # Get feature columns in same order as feature_names
    if feature_names is not None:
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features in training data: {missing_features}")
        X = df[feature_names].values
    else:
        X = df.drop([ch_col, 'longitude', 'latitude'], axis=1, errors='ignore').values
    
    return X, y

def load_prediction_data(stack_path: str, mask_path: Optional[str] = None, feature_names: Optional[list] = None) -> Tuple[np.ndarray, rasterio.DatasetReader]:
    """
    Load prediction data from stack TIF and optionally apply forest mask.
    
    Args:
        stack_path: Path to stack TIF file
        mask_path: Optional path to forest mask TIF
        feature_names: Optional list of feature names for filtering bands
        
    Returns:
        X: Feature matrix for prediction
        src: Rasterio dataset for writing results
    """
    if feature_names is None:
        raise ValueError("feature_names must be provided to ensure consistent features between training and prediction")
    # Read stack file
    with rasterio.open(stack_path) as src:
        stack = src.read()
        stack_crs = src.crs
        
        # Get band descriptions if available
        band_descriptions = src.descriptions
        
        # Filter bands based on feature names if provided
        # Create a mapping of band descriptions to indices
        band_indices = []
        for i, desc in enumerate(band_descriptions):
            if desc in feature_names:
                band_indices.append(i)
        
        if len(band_indices) != len(feature_names):
            missing_features = set(feature_names) - set(band_descriptions)
            raise ValueError(f"Could not find all feature names in stack bands. Missing features: {missing_features}")
        
        # Select only the bands that match feature names
        stack = stack[band_indices]
        
        # Reshape stack to 2D array (bands x pixels)
        n_bands, height, width = stack.shape
        X = stack.reshape(n_bands, -1).T
        
        # Apply mask if provided
        if mask_path:
            with rasterio.open(mask_path) as mask_src:
                # Check CRS
                if mask_src.crs != stack_crs:
                    raise ValueError(f"CRS mismatch: stack {stack_crs} != mask {mask_src.crs}")
                
                # Check dimensions
                if mask_src.shape != (height, width):
                    raise ValueError(f"Shape mismatch: stack {(height, width)} != mask {mask_src.shape}")
                
                mask = mask_src.read(1)
                mask = mask.reshape(-1)
                X = X[mask == 1]
        
        src_copy = rasterio.open(stack_path)
        return X, src_copy

def save_metrics_and_importance(metrics: dict, importance_data: dict, output_dir: str) -> None:
    """
    Save training metrics and feature importance to JSON file, ensuring all values are JSON serializable.
    """
    # Convert any non-serializable values to Python native types
    serializable_metrics = {}
    for key, value in metrics.items():
        if hasattr(value, 'item'):  # Handle numpy/torch numbers
            serializable_metrics[key] = value.item()
        else:
            serializable_metrics[key] = float(value)
    
    serializable_importance = {}
    for key, value in importance_data.items():
        if hasattr(value, 'item'):  # Handle numpy/torch numbers
            serializable_importance[key] = value.item()
        else:
            serializable_importance[key] = float(value)
    """
    Save training metrics and feature importance to JSON file.
    
    Args:
        metrics: Dictionary of training metrics
        importance_data: Dictionary of feature importance data
        output_dir: Directory to save JSON file
    """
    import json
    from pathlib import Path
    
    # Combine metrics and importance data
    output_data = {
        "train_metrics": serializable_metrics,
        "feature_importance": serializable_importance
    }
    
    # Create output path
    output_path = Path(output_dir) / "model_evaluation.json"
    
    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Saved model evaluation data to: {output_path}")

def train_3d_unet(patch_data: List[Tuple[np.ndarray, np.ndarray, str]], 
                 model_params: Dict = None, 
                 training_params: Dict = None) -> Tuple[object, dict]:
    """
    Train 3D U-Net model on patch data.
    
    Args:
        patch_data: List of (features, gedi_target, patch_name) tuples
        model_params: Model hyperparameters
        training_params: Training hyperparameters
        
    Returns:
        Trained model and metrics
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for 3D U-Net training")
    
    # Default parameters
    if model_params is None:
        model_params = {'base_channels': 64}
    if training_params is None:
        training_params = {
            'epochs': 100,
            'batch_size': 4,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'huber_delta': 1.0,
            'shift_radius': 1
        }
    
    # Split data into train/val
    train_size = int(0.8 * len(patch_data))
    train_patches = patch_data[:train_size]
    val_patches = patch_data[train_size:]
    
    print(f"Training on {len(train_patches)} patches, validating on {len(val_patches)} patches")
    
    # Get input dimensions from first patch
    n_bands = patch_data[0][0].shape[0]
    
    # Initialize model
    model = create_3d_unet(
        in_channels=n_bands,
        n_classes=1,
        base_channels=model_params['base_channels']
    )
    
    # Set up training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=training_params['learning_rate'],
        weight_decay=training_params['weight_decay']
    )
    
    best_val_loss = float('inf')
    metrics = {}
    
    # Training loop
    for epoch in tqdm(range(training_params['epochs']), desc="Training epochs"):
        model.train()
        train_loss = 0.0
        
        # Training
        for features, gedi_target, _ in train_patches:
            # Convert to tensors and add batch dimension
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            target_tensor = torch.FloatTensor(gedi_target).unsqueeze(0).to(device)
            
            # Add temporal dimension for 3D conv (treating as single time step)
            features_tensor = features_tensor.unsqueeze(2)  # [1, bands, 1, H, W]
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(features_tensor)
            
            # Compute loss with spatial shift awareness
            loss = modified_huber_loss(
                pred, target_tensor.unsqueeze(1),  # Add channel dim
                delta=training_params['huber_delta'],
                shift_radius=training_params['shift_radius']
            )
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        if val_patches:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for features, gedi_target, _ in val_patches:
                    features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(2).to(device)
                    target_tensor = torch.FloatTensor(gedi_target).unsqueeze(0).to(device)
                    
                    pred = model(features_tensor)
                    loss = modified_huber_loss(
                        pred, target_tensor.unsqueeze(1),
                        delta=training_params['huber_delta'],
                        shift_radius=training_params['shift_radius']
                    )
                    val_loss += loss.item()
            
            val_loss /= len(val_patches)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                metrics = {
                    'train_loss': train_loss / len(train_patches),
                    'val_loss': val_loss,
                    'epoch': epoch
                }
    
    return model, metrics

def train_model(X: np.ndarray, y: np.ndarray, model_type: str = 'rf', batch_size: int = 64,
                test_size: float = 0.2, feature_names: Optional[list] = None,
                n_bands: Optional[int] = None, **kwargs) -> Tuple[object, dict, dict]:
    """
    Train model with optional validation split.
    
    Args:
        X: Feature matrix
        y: Target variable
        model_type: Type of model ('rf', 'mlp', or '3d_unet')
        batch_size: Batch size for neural network training
        test_size: Proportion of data to use for validation
        feature_names: Optional list of feature names
        
    Returns:
        Trained model, training metrics, and feature importance/weights
    """
    # Handle 3D U-Net case
    if model_type == '3d_unet':
        if 'patch_data' not in kwargs:
            raise ValueError("patch_data required for 3D U-Net training")
        model, train_metrics = train_3d_unet(
            kwargs['patch_data'],
            kwargs.get('model_params'),
            kwargs.get('training_params')
        )
        return model, train_metrics, {}
    
    # Split data for traditional models
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    if model_type == 'rf':
        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=5,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_val)
        train_metrics = calculate_metrics(y_pred, y_val)
        
        # Get feature importance
        importance = model.feature_importances_
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        importance_data = {
            name: float(imp) for name, imp in zip(feature_names, importance)
        }
        
    else:  # MLP model
        # Create normalized dataloaders
        train_loader, val_loader, scaler_mean, scaler_std = create_normalized_dataloader(
            X_train, X_val, y_train, y_val, batch_size=batch_size, n_bands=n_bands
        )
        
        # Initialize model
        model = MLPRegressionModel(input_size=X.shape[1])
        if torch.cuda.is_available():
            model = model.cuda()
            
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())
        num_epochs = 100
        best_val_loss = float('inf')
        
        # Training loop with tqdm progress bar
        for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
            model.train()
            for batch_X, batch_y in train_loader:
                if torch.cuda.is_available():
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_predictions = []
            val_targets = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    if torch.cuda.is_available():
                        batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                    outputs = model(batch_X)
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            val_metrics = calculate_metrics(val_predictions, val_targets)
            val_loss = val_metrics['RMSE']
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                train_metrics = val_metrics
        
        # Get feature importance (using weights of first layer as proxy)
        with torch.no_grad():
            weights = model.layers[0].weight.abs().mean(dim=0).cpu().numpy()
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(weights))]
            importance_data = {}
            for name, weight in zip(feature_names, weights):
                # Convert numpy.float32 to Python float
                weight_value = weight.item() if hasattr(weight, 'item') else float(weight)
                importance_data[name] = weight_value
        
        # Store normalization parameters with model
        model.scaler_mean = scaler_mean
        model.scaler_std = scaler_std
    
    # Sort importance by value
    importance_data = dict(sorted(importance_data.items(), key=lambda x: x[1], reverse=True))
    
    # Print metrics and top features
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.3f}")
    
    print("\nTop 5 Important Features:")
    for name, imp in list(importance_data.items())[:5]:
        print(f"{name}: {imp:.3f}")
    
    return model, train_metrics, importance_data

def save_predictions(predictions: np.ndarray, src: rasterio.DatasetReader, output_path: str,
                    mask_path: Optional[str] = None) -> None:
    """
    Save predictions to a GeoTIFF file.
    
    Args:
        predictions: Model predictions
        src: Source rasterio dataset for metadata
        output_path: Path to save predictions
        mask_path: Optional path to forest mask TIF
    """
    # Create output profile
    profile = src.profile.copy()
    profile.update(count=1, dtype='float32')
    
    # Initialize prediction array
    height, width = src.height, src.width
    pred_array = np.zeros((height, width), dtype='float32')
    
    if mask_path:
        # Apply predictions only to masked areas
        with rasterio.open(mask_path) as mask_src:
            # Check CRS
            if mask_src.crs != src.crs:
                raise ValueError(f"CRS mismatch: source {src.crs} != mask {mask_src.crs}")
            
            mask = mask_src.read(1)
            mask_idx = np.where(mask.reshape(-1) == 1)[0]
            pred_array.reshape(-1)[mask_idx] = predictions
    else:
        # Apply predictions to all pixels
        pred_array = predictions.reshape(height, width)
    
    try:
        # Save predictions
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(pred_array, 1)
    finally:
        src.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Train model and generate canopy height predictions')
    
    # Input paths
    parser.add_argument('--training-data', type=str, required=False,
                       help='Path to training data CSV (not used for 3D U-Net)')
    parser.add_argument('--stack', type=str, required=False,
                       help='Path to stack TIF file (not used for 3D U-Net)')
    parser.add_argument('--mask', type=str, required=False,
                       help='Path to forest mask TIF')
    parser.add_argument('--buffered-mask', type=str, required=False,
                       help='Path to buffered forest mask TIF')
    
    # 3D U-Net specific paths
    parser.add_argument('--patches-dir', type=str, default='chm_outputs',
                       help='Directory containing patch TIF files')
    parser.add_argument('--patch-pattern', type=str, default='*_patch*.tif',
                       help='Pattern to match patch files')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='chm_outputs',
                       help='Output directory for predictions')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='rf', choices=['rf', 'mlp', '3d_unet'],
                       help='Model type: random forest (rf), MLP neural network (mlp), or 3D U-Net (3d_unet)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for neural network training')
    parser.add_argument('--n-bands', type=int, default=None,
                       help='Number of spectral bands for band-wise normalization')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data to use for validation')
    parser.add_argument('--apply-forest-mask', action='store_true',
                       help='Apply forest mask to predictions')
    parser.add_argument('--ch_col', type=str, default='rh',
                       help='Column name for canopy height')
    
    # 3D U-Net specific parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs for 3D U-Net')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate for 3D U-Net')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay for 3D U-Net')
    parser.add_argument('--huber-delta', type=float, default=1.0,
                       help='Huber loss delta parameter')
    parser.add_argument('--shift-radius', type=int, default=1,
                       help='Spatial shift radius for GEDI alignment')
    parser.add_argument('--base-channels', type=int, default=64,
                       help='Base number of channels in 3D U-Net')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    ch_col = args.ch_col
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.model == '3d_unet':
        # 3D U-Net workflow
        print("Loading patch data for 3D U-Net...")
        
        # Load patch data
        patch_data = load_patches_from_directory(args.patches_dir, args.patch_pattern)
        
        if not patch_data:
            raise ValueError(f"No patch files found in {args.patches_dir} with pattern {args.patch_pattern}")
        
        print(f"Loaded {len(patch_data)} patches")
        
        # Set up training parameters
        model_params = {'base_channels': args.base_channels}
        training_params = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'huber_delta': args.huber_delta,
            'shift_radius': args.shift_radius
        }
        
        # Train 3D U-Net
        print("Training 3D U-Net model...")
        model, train_metrics, importance_data = train_model(
            None, None,  # X, y not used for 3D U-Net
            model_type=args.model,
            patch_data=patch_data,
            model_params=model_params,
            training_params=training_params
        )
        
        # Save model
        model_path = os.path.join(args.output_dir, '3d_unet_model.pth')
        if TORCH_AVAILABLE:
            torch.save(model.state_dict(), model_path)
            print(f"Saved 3D U-Net model to: {model_path}")
        
        # Save metrics
        save_metrics_and_importance(train_metrics, {}, args.output_dir)
        
        print("3D U-Net training completed!")
        print(f"Final training loss: {train_metrics.get('train_loss', 'N/A'):.4f}")
        print(f"Final validation loss: {train_metrics.get('val_loss', 'N/A'):.4f}")
        
    else:
        # Traditional workflow (RF/MLP)
        if not args.training_data or not args.stack:
            raise ValueError("--training-data and --stack are required for traditional models")
        
        # Load training data
        print("Loading training data...")
        df = pd.read_csv(args.training_data)
        
        # Filter out samples with slope > 20
        slope_cols = [col for col in df.columns if col.endswith('_slope')]
        for col in slope_cols:
            df = df[df[col] <= 20]
        
        # Define columns to remove
        remove_cols = [ch_col, 'longitude', 'latitude', 'rh',
                      'digital_elevation_model', 'digital_elevation_model_srtm', 'elev_lowestmode']
        feature_names = [col for col in df.columns if col not in remove_cols]
        
        # Write filtered DataFrame back to file for consistency
        filtered_training_path = os.path.join(args.output_dir, 'filtered_training.csv')
        df.to_csv(filtered_training_path, index=False)
        
        # Load filtered training data
        X, y = load_training_data(filtered_training_path, args.buffered_mask,
                                 feature_names=feature_names, ch_col=ch_col)
        print(f"Loaded training data with {X.shape[1]} features and {len(y)} samples")
        
        # Train model
        print("Training model...")
        model, train_metrics, importance_data = train_model(
            X, y,
            model_type=args.model,
            batch_size=args.batch_size,
            test_size=args.test_size,
            feature_names=feature_names,
            n_bands=args.n_bands
        )
        
        # Save metrics and importance
        save_metrics_and_importance(train_metrics, importance_data, args.output_dir)
        
        # Load prediction data and generate predictions
        print("Loading prediction data...")
        print(f"Using {len(feature_names)} features for prediction: {', '.join(feature_names)}")
        X_pred, src = load_prediction_data(args.stack, args.mask, feature_names=feature_names)
        print(f"Loaded prediction data with shape: {X_pred.shape}")
        
        if X_pred.shape[1] != X.shape[1]:
            raise ValueError(f"Feature count mismatch: Training has {X.shape[1]} features, but prediction data has {X_pred.shape[1]} features")
        
        # Make predictions
        print("Generating predictions...")
        if args.model == 'rf':
            predictions = model.predict(X_pred)
        else:  # MLP model
            model.eval()
            with torch.no_grad():
                # Normalize prediction data
                X_pred_tensor = torch.FloatTensor(X_pred)
                X_pred_normalized = (X_pred_tensor - model.scaler_mean) / model.scaler_std
                
                # Make predictions in batches
                predictions = []
                for i in range(0, len(X_pred), args.batch_size):
                    batch = X_pred_normalized[i:i + args.batch_size]
                    if torch.cuda.is_available():
                        batch = batch.cuda()
                    pred = model(batch)
                    predictions.extend(pred.cpu().numpy())
                predictions = np.array(predictions)
        
        print(f"Generated {len(predictions)} predictions")
        output_path = Path(args.output_dir) / f"{Path(args.stack).stem.replace('stack_', 'predictCH')}_{ch_col}.tif"
        
        # Save predictions
        print(f"Saving predictions to: {output_path}")
        save_predictions(predictions, src, output_path, args.mask)
        print("Done!")
    
    # Save metrics and importance
    save_metrics_and_importance(train_metrics, importance_data, args.output_dir)
    
    # Load prediction data
    print("Loading prediction data...")
    print(f"Using {len(feature_names)} features for prediction: {', '.join(feature_names)}")
    X_pred, src = load_prediction_data(args.stack, args.mask, feature_names=feature_names)
    print(f"Loaded prediction data with shape: {X_pred.shape}")
    
    if X_pred.shape[1] != X.shape[1]:
        raise ValueError(f"Feature count mismatch: Training has {X.shape[1]} features, but prediction data has {X_pred.shape[1]} features")
    
    # Make predictions
    print("Generating predictions...")
    if args.model == 'rf':
        predictions = model.predict(X_pred)
    else:  # MLP model
        model.eval()
        with torch.no_grad():
            # Normalize prediction data
            X_pred_tensor = torch.FloatTensor(X_pred)
            X_pred_normalized = (X_pred_tensor - model.scaler_mean) / model.scaler_std
            
            # Make predictions in batches
            predictions = []
            for i in range(0, len(X_pred), args.batch_size):
                batch = X_pred_normalized[i:i + args.batch_size]
                if torch.cuda.is_available():
                    batch = batch.cuda()
                pred = model(batch)
                predictions.extend(pred.cpu().numpy())
            predictions = np.array(predictions)
    print(f"Generated {len(predictions)} predictions")
    output_path = Path(args.output_dir) / f"{Path(args.stack).stem.replace('stack_', 'predictCH')}_{ch_col}.tif"
    
    # Save predictions
    # output_path = os.path.join(args.output_dir, output_filename)
    print(f"Saving predictions to: {output_path}")
    save_predictions(predictions, src, output_path, args.mask)
    print("Done!")

if __name__ == "__main__":
    main()