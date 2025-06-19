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

# 2D U-Net Model for non-temporal data
class Height2DUNet(nn.Module):
    """2D U-Net for canopy height prediction from non-temporal patches."""
    
    def __init__(self, in_channels, n_classes=1, base_channels=64):
        super().__init__()
        
        # Encoder
        self.encoder1 = self.conv_block(in_channels, base_channels)
        self.encoder2 = self.conv_block(base_channels, base_channels * 2)
        self.encoder3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.encoder4 = self.conv_block(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.decoder4 = self.upconv_block(base_channels * 16, base_channels * 8)
        self.decoder3 = self.upconv_block(base_channels * 16, base_channels * 4)  # 16 = 8 + 8 from skip
        self.decoder2 = self.upconv_block(base_channels * 8, base_channels * 2)   # 8 = 4 + 4 from skip
        self.decoder1 = self.upconv_block(base_channels * 4, base_channels)       # 4 = 2 + 2 from skip
        
        # Final prediction
        self.final_conv = nn.Conv2d(base_channels, n_classes, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # (B, 64, H, W)
        e2 = self.encoder2(nn.MaxPool2d(2)(e1))  # (B, 128, H/2, W/2)
        e3 = self.encoder3(nn.MaxPool2d(2)(e2))  # (B, 256, H/4, W/4)
        e4 = self.encoder4(nn.MaxPool2d(2)(e3))  # (B, 512, H/8, W/8)
        
        # Bottleneck
        b = self.bottleneck(nn.MaxPool2d(2)(e4))  # (B, 1024, H/16, W/16)
        
        # Decoder with skip connections
        d4 = self.decoder4(b)  # (B, 512, H/8, W/8)
        d4 = torch.cat([d4, e4], dim=1)  # (B, 1024, H/8, W/8)
        
        d3 = self.decoder3(d4)  # (B, 256, H/4, W/4)
        d3 = torch.cat([d3, e3], dim=1)  # (B, 512, H/4, W/4)
        
        d2 = self.decoder2(d3)  # (B, 128, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)  # (B, 256, H/2, W/2)
        
        d1 = self.decoder1(d2)  # (B, 64, H, W)
        
        # Final prediction
        out = self.final_conv(d1)  # (B, 1, H, W)
        
        return out.squeeze(1)  # (B, H, W)

def create_2d_unet(in_channels: int, n_classes: int = 1, base_channels: int = 64):
    """Create 2D U-Net model."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for 2D U-Net")
    return Height2DUNet(in_channels, n_classes, base_channels)
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
    Load patch data from GeoTIFF file with improved temporal and normalization support.
    
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
        
        # Crop to 256x256 if needed (handle 257x257 patches)
        if features.shape[1] > 256 or features.shape[2] > 256:
            print(f"Cropping patch from {features.shape[1]}x{features.shape[2]} to 256x256")
            features = features[:, :256, :256]
            gedi_target = gedi_target[:256, :256]
        
        # Apply improved normalization with temporal support
        if normalize_bands:
            features = apply_band_normalization(features, band_descriptions, feature_indices)
    
    return features, gedi_target, band_info

def apply_band_normalization(features: np.ndarray, band_descriptions: list, feature_indices: list) -> np.ndarray:
    """Apply band-specific normalization with temporal support."""
    
    for i, idx in enumerate(feature_indices):
        desc = band_descriptions[idx]
        if not desc:
            continue
            
        # Handle temporal bands (with _M## suffix)
        base_desc = desc.split('_M')[0] if '_M' in desc else desc
        
        # Apply normalization based on base description
        if base_desc.startswith('S1_'):
            # Sentinel-1 normalization: (val + 25) / 25
            features[i] = (features[i] + 25) / 25
        elif base_desc in ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']:
            # Sentinel-2 reflectance: val / 10000, clip to [0,1]
            features[i] = np.clip(features[i] / 10000.0, 0, 1)
        elif base_desc == 'NDVI':
            # NDVI: clip to [-1, 1]
            features[i] = np.clip(features[i], -1, 1)
        elif 'elevation' in desc.lower():
            features[i] = normalize_srtm_elevation(features[i])
        elif 'slope' in desc.lower():
            features[i] = normalize_srtm_slope(features[i])
        elif 'aspect' in desc.lower():
            features[i] = normalize_srtm_aspect(features[i])
        elif base_desc.startswith('ALOS2_'):
            # ALOS2: keep as-is or apply light normalization if needed
            features[i] = features[i]  # No normalization for now
        elif base_desc.startswith('ch_'):
            features[i] = normalize_canopy_height(features[i])
        
        # Replace any remaining NaN/inf values
        features[i] = np.nan_to_num(features[i], nan=0.0, posinf=0.0, neginf=0.0)
    
    return features

def extract_sparse_gedi_pixels(features: np.ndarray, gedi_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature vectors only for pixels with valid GEDI data.
    
    Args:
        features: Feature array [bands, height, width]
        gedi_target: GEDI target array [height, width]
        
    Returns:
        X: Feature matrix [n_valid_pixels, n_bands]
        y: Target vector [n_valid_pixels]
    """
    # Find valid GEDI pixels (not NaN and > 0)
    valid_mask = ~np.isnan(gedi_target) & (gedi_target > 0)
    valid_indices = np.where(valid_mask)
    
    if len(valid_indices[0]) == 0:
        raise ValueError("No valid GEDI pixels found in patch")
    
    # Extract features for valid pixels only
    X = features[:, valid_indices[0], valid_indices[1]].T  # Shape: (n_pixels, n_bands)
    y = gedi_target[valid_indices]  # Shape: (n_pixels,)
    
    print(f"Extracted {len(y)} valid GEDI pixels from {gedi_target.size} total pixels ({len(y)/gedi_target.size*100:.2f}%)")
    
    return X, y

def detect_temporal_mode(band_descriptions: list) -> bool:
    """
    Detect if patch data is temporal based on band naming convention.
    
    Args:
        band_descriptions: List of band descriptions
        
    Returns:
        True if temporal data detected, False otherwise
    """
    temporal_indicators = ['_M01', '_M02', '_M03', '_M04', '_M05', '_M06',
                          '_M07', '_M08', '_M09', '_M10', '_M11', '_M12']
    
    for desc in band_descriptions:
        if desc and any(indicator in desc for indicator in temporal_indicators):
            return True
    
    return False

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

def train_2d_unet(patch_path: str, model_params: Dict = None, training_params: Dict = None) -> Tuple[object, dict]:
    """
    Train 2D U-Net model on single patch with data augmentation.
    
    Args:
        patch_path: Path to patch TIF file
        model_params: Model hyperparameters
        training_params: Training hyperparameters
        
    Returns:
        Trained model and metrics
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for 2D U-Net training")
    
    # Default parameters
    if model_params is None:
        model_params = {'base_channels': 32}  # Smaller for 2D
    if training_params is None:
        training_params = {
            'epochs': 50,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'huber_delta': 1.0,
            'shift_radius': 1
        }
    
    print(f"Loading patch data for 2D U-Net training...")
    features, gedi_target, band_info = load_patch_data(patch_path)
    
    # Resize to 256x256 if needed
    if features.shape[1] != 256 or features.shape[2] != 256:
        from scipy.ndimage import zoom
        scale_h = 256 / features.shape[1]
        scale_w = 256 / features.shape[2]
        
        resized_features = np.zeros((features.shape[0], 256, 256), dtype=features.dtype)
        for i in range(features.shape[0]):
            resized_features[i] = zoom(features[i], (scale_h, scale_w), order=1)
        
        resized_gedi = zoom(gedi_target, (scale_h, scale_w), order=0)
        features, gedi_target = resized_features, resized_gedi
    
    n_bands = features.shape[0]
    print(f"Training 2D U-Net on {n_bands} bands, patch size: {features.shape[1]}x{features.shape[2]}")
    
    # Initialize model
    model = create_2d_unet(
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
    
    # Convert to tensors
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)  # [1, bands, H, W]
    target_tensor = torch.FloatTensor(gedi_target).unsqueeze(0).to(device)  # [1, H, W]
    
    # Create mask for valid GEDI pixels
    valid_mask = ~torch.isnan(target_tensor) & (target_tensor > 0)
    
    best_loss = float('inf')
    metrics = {}
    
    # Training loop
    for epoch in tqdm(range(training_params['epochs']), desc="Training 2D U-Net"):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(features_tensor)  # [1, H, W]
        
        # Compute loss only on valid GEDI pixels
        if valid_mask.sum() > 0:
            valid_pred = pred[valid_mask]
            valid_target = target_tensor[valid_mask]
            loss = nn.MSELoss()(valid_pred, valid_target)
        else:
            loss = torch.tensor(0.0, requires_grad=True, device=device)
        
        # Backward pass
        if loss.item() > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            metrics = {
                'train_loss': current_loss,
                'epoch': epoch,
                'valid_gedi_pixels': valid_mask.sum().item()
            }
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{training_params['epochs']}: Loss = {current_loss:.4f}")
    
    print(f"2D U-Net training complete. Best loss: {best_loss:.4f}")
    return model, metrics

def train_3d_unet(patch_path: str, model_params: Dict = None, training_params: Dict = None) -> Tuple[object, dict]:
    """
    Train 3D U-Net model on temporal patch with data augmentation.
    
    Args:
        patch_path: Path to temporal patch TIF file
        model_params: Model hyperparameters
        training_params: Training hyperparameters
        
    Returns:
        Trained model and metrics
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for 3D U-Net training")
    
    # Default parameters
    if model_params is None:
        model_params = {'base_channels': 32}  # Smaller for memory
    if training_params is None:
        training_params = {
            'epochs': 30,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'huber_delta': 1.0,
            'shift_radius': 1
        }
    
    print(f"Loading temporal patch data for 3D U-Net training...")
    features, gedi_target, band_info = load_patch_data(patch_path)
    
    # Import the improved temporal dataset from our previous work
    from train_temporal_fixed import ImprovedTemporalDataset, MaskedTemporalUNet
    
    # Use the improved temporal dataset
    dataset = ImprovedTemporalDataset(patch_path, patch_size=256, augment=True)
    
    # Initialize model
    model = MaskedTemporalUNet(in_channels=15, n_classes=1).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=training_params['learning_rate'],
        weight_decay=training_params['weight_decay']
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_loss = float('inf')
    metrics = {}
    
    # Training loop
    for epoch in tqdm(range(training_params['epochs']), desc="Training 3D U-Net"):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Train on augmented patches
        for i in range(len(dataset)):
            features, target, gedi_mask, availability = dataset[i]
            
            features = features.unsqueeze(0).to(device)  # Add batch dim
            target = target.unsqueeze(0).to(device)
            gedi_mask = gedi_mask.unsqueeze(0).to(device)
            availability = availability.unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(features, availability)
            
            # Compute loss only on valid GEDI pixels
            if gedi_mask.sum() > 0:
                valid_pred = pred[gedi_mask > 0]
                valid_target = target[gedi_mask > 0]
                loss = nn.MSELoss()(valid_pred, valid_target)
            else:
                loss = torch.tensor(0.0, requires_grad=True, device=device)
            
            # Backward pass
            if loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        if avg_loss < best_loss:
            best_loss = avg_loss
            metrics = {
                'train_loss': avg_loss,
                'epoch': epoch
            }
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{training_params['epochs']}: Loss = {avg_loss:.4f}")
    
    print(f"3D U-Net training complete. Best loss: {best_loss:.4f}")
    return model, metrics

def train_model(X: np.ndarray, y: np.ndarray, model_type: str = 'rf', 
                batch_size: int = 64, test_size: float = 0.2, feature_names: Optional[list] = None,
                n_bands: Optional[int] = None) -> Tuple[object, dict, dict]:
    """
    Training function for traditional models (RF/MLP only).
    U-Net models are handled separately in main().
    
    Args:
        X: Feature matrix (extracted from patch for GEDI pixels)
        y: Target variable (extracted from patch for GEDI pixels)
        model_type: Type of model ('rf' or 'mlp')
        batch_size: Batch size for MLP training
        test_size: Proportion of data to use for validation
        feature_names: Optional list of feature names
        
    Returns:
        Trained model, training metrics, and feature importance/weights
    """
    if model_type not in ['rf', 'mlp']:
        raise ValueError(f"train_model only supports 'rf' and 'mlp', got '{model_type}'")
    
    if X.shape[0] < 10:  # Need minimum samples for train/val split
        raise ValueError(f"Insufficient GEDI pixels ({X.shape[0]}) for {model_type.upper()} training. Need at least 10.")
    
    print(f"Training {model_type.upper()} on {X.shape[0]} GEDI pixels with {X.shape[1]} features")
    
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

def train_2d_unet(features: np.ndarray, gedi_target: np.ndarray, 
                  epochs: int = 50, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                  base_channels: int = 32, huber_delta: float = 1.0, shift_radius: int = 1) -> Tuple[nn.Module, Dict]:
    """
    Train 2D U-Net model on patch data.
    
    Args:
        features: Feature array [bands, height, width]
        gedi_target: GEDI target array [height, width]
        epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        base_channels: Base channels for U-Net
        huber_delta: Huber loss delta
        shift_radius: Spatial shift radius for GEDI alignment
        
    Returns:
        model: Trained 2D U-Net model
        metrics: Training metrics dictionary
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for 2D U-Net training")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training 2D U-Net on device: {device}")
    
    # Create model
    model = create_2d_unet(in_channels=features.shape[0], n_classes=1, base_channels=base_channels)
    model = model.to(device)
    
    # Setup optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Convert data to tensors
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)  # Add batch dim
    gedi_tensor = torch.FloatTensor(gedi_target).unsqueeze(0).to(device)   # Add batch dim
    
    # Create valid mask for GEDI pixels
    valid_mask = ~torch.isnan(gedi_tensor) & (gedi_tensor > 0)
    
    print(f"Valid GEDI pixels: {valid_mask.sum().item()}/{gedi_tensor.numel()}")
    
    # Training loop
    model.train()
    train_losses = []
    
    for epoch in tqdm(range(epochs), desc="Training 2D U-Net"):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(features_tensor)  # [1, H, W]
        
        # Calculate modified Huber loss with shift awareness
        loss = modified_huber_loss(predictions, gedi_tensor, valid_mask, huber_delta, shift_radius)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    
    # Calculate final metrics on valid GEDI pixels
    model.eval()
    with torch.no_grad():
        final_predictions = model(features_tensor)
        
        # Extract predictions and targets for valid GEDI pixels only
        valid_preds = final_predictions[valid_mask].cpu().numpy()
        valid_targets = gedi_tensor[valid_mask].cpu().numpy()
        
        # Calculate metrics using existing function
        metrics = calculate_metrics(valid_preds, valid_targets)
        metrics['train_loss'] = np.mean(train_losses[-10:])  # Average of last 10 epochs
        metrics['final_loss'] = train_losses[-1]
    
    return model, metrics

def separate_temporal_nontemporal_bands(features: np.ndarray, band_descriptions: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate temporal and non-temporal bands from feature array.
    
    Args:
        features: Feature array [bands, height, width]
        band_descriptions: List of band descriptions
        
    Returns:
        temporal_features: Temporal bands [temporal_bands, height, width]
        nontemporal_features: Non-temporal bands [nontemporal_bands, height, width]
    """
    temporal_indices = []
    nontemporal_indices = []
    
    for i, desc in enumerate(band_descriptions):
        if desc and desc not in ['rh', 'forest_mask']:
            # Check if band has monthly suffix (_M01 to _M12)
            if any(f'_M{m:02d}' in desc for m in range(1, 13)):
                temporal_indices.append(i)
            else:
                nontemporal_indices.append(i)
    
    temporal_features = features[temporal_indices] if temporal_indices else np.empty((0, features.shape[1], features.shape[2]))
    nontemporal_features = features[nontemporal_indices] if nontemporal_indices else np.empty((0, features.shape[1], features.shape[2]))
    
    print(f"Separated bands: {len(temporal_indices)} temporal, {len(nontemporal_indices)} non-temporal")
    
    return temporal_features, nontemporal_features

def train_3d_unet(features: np.ndarray, gedi_target: np.ndarray,
                  epochs: int = 50, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                  base_channels: int = 32, huber_delta: float = 1.0, shift_radius: int = 1) -> Tuple[nn.Module, Dict]:
    """
    Train 3D U-Net model on temporal patch data.
    
    Args:
        features: Feature array [bands, height, width] (all bands including non-temporal)
        gedi_target: GEDI target array [height, width]
        epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        base_channels: Base channels for U-Net
        huber_delta: Huber loss delta
        shift_radius: Spatial shift radius for GEDI alignment
        
    Returns:
        model: Trained 3D U-Net model
        metrics: Training metrics dictionary
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for 3D U-Net training")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training 3D U-Net on device: {device}")
    
    # Load band descriptions to separate temporal/non-temporal bands
    # For now, assume the temporal structure: 180 temporal bands (15 bands × 12 months)
    n_bands = features.shape[0]
    h, w = features.shape[1], features.shape[2]
    
    # Expected structure: 180 temporal + 14 non-temporal = 194 total feature bands
    n_temporal_expected = 180  # 15 bands × 12 months
    n_nontemporal_expected = n_bands - n_temporal_expected
    
    if n_bands >= n_temporal_expected:
        # Separate temporal and non-temporal bands
        temporal_features = features[:n_temporal_expected]  # First 180 bands
        nontemporal_features = features[n_temporal_expected:] if n_bands > n_temporal_expected else np.empty((0, h, w))
        
        print(f"Using expected temporal structure: {n_temporal_expected} temporal + {len(nontemporal_features)} non-temporal bands")
        
        # Reshape temporal data: (180, h, w) -> (15, 12, h, w)
        n_channels_per_month = 15  # S1(2) + S2(11) + ALOS2(2) = 15
        n_months = 12
        
        temporal_features_reshaped = temporal_features.reshape(n_channels_per_month, n_months, h, w)
        
        # Handle missing values in temporal data
        temporal_features_reshaped = np.nan_to_num(temporal_features_reshaped, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Reshaped temporal data: {n_temporal_expected} bands -> {n_channels_per_month} channels × {n_months} months")
        
        # For 3D U-Net, we'll use temporal features. Non-temporal can be added as extra channels if needed
        if len(nontemporal_features) > 0:
            # Repeat non-temporal features across time dimension
            nontemporal_expanded = np.tile(nontemporal_features[:, np.newaxis, :, :], (1, n_months, 1, 1))
            # Combine temporal and non-temporal
            combined_features = np.concatenate([temporal_features_reshaped, nontemporal_expanded], axis=0)
            n_total_channels = n_channels_per_month + len(nontemporal_features)
        else:
            combined_features = temporal_features_reshaped
            n_total_channels = n_channels_per_month
    else:
        raise ValueError(f"Insufficient bands for temporal processing: got {n_bands}, expected at least {n_temporal_expected}")
    
    print(f"Final 3D input shape: {n_total_channels} channels × {n_months} months × {h}×{w}")
    
    # Create model - use smaller base_channels to avoid memory issues and temporal dimension problems
    # For temporal data with only 12 months, we need to be careful with downsampling
    model_base_channels = min(base_channels, 16)  # Use smaller channels for 3D
    print(f"Creating 3D U-Net with {model_base_channels} base channels (reduced for temporal processing)")
    
    try:
        model = create_3d_unet(in_channels=n_total_channels, n_classes=1, base_channels=model_base_channels)
        model = model.to(device)
        
        # Test the model with a small input to verify it works
        test_input = torch.randn(1, n_total_channels, n_months, 32, 32).to(device)
        with torch.no_grad():
            test_output = model(test_input)
            print(f"Model test successful: {test_input.shape} -> {test_output.shape}")
    except Exception as e:
        print(f"Model creation failed with error: {e}")
        print("Falling back to simplified temporal processing...")
        
        # Fallback: reduce temporal dimension by averaging or use 2D approach
        # Average across temporal dimension to create 2D input
        averaged_features = np.mean(combined_features, axis=1)  # Average across time
        print(f"Fallback: Using temporal average, shape: {averaged_features.shape}")
        
        # Use 2D U-Net instead
        model = create_2d_unet(in_channels=averaged_features.shape[0], n_classes=1, base_channels=base_channels)
        model = model.to(device)
        
        # Update the combined_features for training
        combined_features = averaged_features
        print("Using 2D U-Net with temporally averaged features as fallback")
    
    # Setup optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Convert data to tensors - handle both 3D and 2D cases
    if len(combined_features.shape) == 4:  # 3D case: (channels, time, h, w)
        features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(device)  # Add batch dim
    else:  # 2D fallback case: (channels, h, w)
        features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(device)  # Add batch dim
    
    gedi_tensor = torch.FloatTensor(gedi_target).unsqueeze(0).to(device)           # Add batch dim
    
    # Create valid mask for GEDI pixels
    valid_mask = ~torch.isnan(gedi_tensor) & (gedi_tensor > 0)
    
    print(f"Valid GEDI pixels: {valid_mask.sum().item()}/{gedi_tensor.numel()}")
    
    # Training loop
    model.train()
    train_losses = []
    
    for epoch in tqdm(range(epochs), desc="Training 3D U-Net"):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(features_tensor)  # [1, H, W]
        
        # Calculate modified Huber loss with shift awareness
        loss = modified_huber_loss(predictions, gedi_tensor, valid_mask, huber_delta, shift_radius)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    
    # Calculate final metrics on valid GEDI pixels
    model.eval()
    with torch.no_grad():
        final_predictions = model(features_tensor)
        
        # Extract predictions and targets for valid GEDI pixels only
        valid_preds = final_predictions[valid_mask].cpu().numpy()
        valid_targets = gedi_tensor[valid_mask].cpu().numpy()
        
        # Calculate metrics using existing function
        metrics = calculate_metrics(valid_preds, valid_targets)
        metrics['train_loss'] = np.mean(train_losses[-10:])  # Average of last 10 epochs
        metrics['final_loss'] = train_losses[-1]
    
    return model, metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Unified patch-based training for all model types')
    
    # Primary input (required for all models)
    parser.add_argument('--patch-path', type=str, required=True,
                       help='Path to patch TIF file with GEDI rh band')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='chm_outputs',
                       help='Output directory for models and predictions')
    
    # Model selection
    parser.add_argument('--model', type=str, default='rf', choices=['rf', 'mlp', '2d_unet', '3d_unet'],
                       help='Model type: random forest (rf), MLP (mlp), 2D U-Net (2d_unet), or 3D U-Net (3d_unet)')
    
    # Traditional model parameters (RF/MLP)
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of GEDI pixels to use for validation (RF/MLP only)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    
    # Neural network parameters (all U-Nets)
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (U-Net models)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate (U-Net models)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (U-Net models)')
    parser.add_argument('--base-channels', type=int, default=32,
                       help='Base number of channels in U-Net models')
    
    # Advanced training parameters
    parser.add_argument('--huber-delta', type=float, default=1.0,
                       help='Huber loss delta parameter (U-Net models)')
    parser.add_argument('--shift-radius', type=int, default=1,
                       help='Spatial shift radius for GEDI alignment (U-Net models)')
    
    # Generation and evaluation
    parser.add_argument('--generate-prediction', action='store_true',
                       help='Generate prediction map after training')
    parser.add_argument('--prediction-output', type=str, default=None,
                       help='Output path for prediction TIF (auto-generated if not specified)')
    
    return parser.parse_args()

def main():
    """Unified main function for all model types using patch-based input."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Training {args.model.upper()} model using patch: {args.patch_path}")
    
    # Load patch data
    print("Loading patch data...")
    features, gedi_target, band_info = load_patch_data(args.patch_path, normalize_bands=True)
    
    # Detect temporal mode
    band_descriptions = list(band_info.keys())
    is_temporal = detect_temporal_mode(band_descriptions)
    
    print(f"Patch data: {features.shape[0]} bands, {features.shape[1]}x{features.shape[2]} pixels")
    print(f"Temporal mode detected: {is_temporal}")
    print(f"Valid GEDI pixels: {np.sum(~np.isnan(gedi_target) & (gedi_target > 0))}/{gedi_target.size}")
    
    # Validate model-data compatibility
    if args.model == '2d_unet' and is_temporal:
        raise ValueError("2D U-Net cannot be used with temporal data. Use '3d_unet' or enable non-temporal mode in chm_main.py")
    if args.model == '3d_unet' and not is_temporal:
        raise ValueError("3D U-Net requires temporal data. Use '2d_unet' or enable temporal mode in chm_main.py")
    
    # Train model based on type
    if args.model in ['rf', 'mlp']:
        # Extract sparse GEDI pixels for traditional models
        print("Extracting sparse GEDI pixels for RF/MLP training...")
        X, y = extract_sparse_gedi_pixels(features, gedi_target)
        
        # Create feature names from band descriptions
        feature_names = [desc for desc in band_descriptions if desc and desc not in ['rh', 'forest_mask']]
        
        # Train traditional model
        model, train_metrics, importance_data = train_model(
            X, y,
            model_type=args.model,
            batch_size=args.batch_size,
            test_size=args.test_size,
            feature_names=feature_names
        )
        
        # Save model
        if args.model == 'rf':
            import joblib
            model_path = os.path.join(args.output_dir, 'rf_model.pkl')
            joblib.dump(model, model_path)
        else:  # MLP
            model_path = os.path.join(args.output_dir, 'mlp_model.pth')
            if TORCH_AVAILABLE:
                torch.save(model.state_dict(), model_path)
        
        print(f"Saved {args.model.upper()} model to: {model_path}")
        
    elif args.model == '2d_unet':
        # Train 2D U-Net
        print("Training 2D U-Net...")
        model, train_metrics = train_2d_unet(
            features, gedi_target,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            base_channels=args.base_channels,
            huber_delta=args.huber_delta,
            shift_radius=args.shift_radius
        )
        
        # Save model
        model_path = os.path.join(args.output_dir, '2d_unet_model.pth')
        if TORCH_AVAILABLE:
            torch.save(model.state_dict(), model_path)
        print(f"Saved 2D U-Net model to: {model_path}")
        
        importance_data = {}  # U-Net doesn't have traditional feature importance
        
    elif args.model == '3d_unet':
        # Train 3D U-Net
        print("Training 3D U-Net...")
        model, train_metrics = train_3d_unet(
            features, gedi_target,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            base_channels=args.base_channels,
            huber_delta=args.huber_delta,
            shift_radius=args.shift_radius
        )
        
        # Save model
        model_path = os.path.join(args.output_dir, '3d_unet_model.pth')
        if TORCH_AVAILABLE:
            torch.save(model.state_dict(), model_path)
        print(f"Saved 3D U-Net model to: {model_path}")
        
        importance_data = {}  # U-Net doesn't have traditional feature importance
    
    # Save metrics and importance
    metrics_path = os.path.join(args.output_dir, 'training_metrics.json')
    import json
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    with open(metrics_path, 'w') as f:
        json.dump({
            'metrics': convert_numpy_types(train_metrics),
            'importance': convert_numpy_types(importance_data),
            'model_type': args.model,
            'temporal_mode': is_temporal,
            'patch_path': args.patch_path
        }, f, indent=2)
    
    print(f"Saved training metrics to: {metrics_path}")
    
    # Print training results
    print("\\nTraining Results:")
    for metric, value in train_metrics.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    if importance_data:
        print("\\nTop 5 Important Features:")
        for name, imp in list(importance_data.items())[:5]:
            print(f"{name}: {imp:.3f}")
    
    # Generate prediction if requested or automatically for RF/MLP
    if args.generate_prediction or args.model in ['rf', 'mlp']:
        print("\\nGenerating prediction map...")
        
        if args.model in ['rf', 'mlp']:
            # For traditional models, predict on all pixels
            print("Reshaping patch for pixel-wise prediction...")
            # Reshape features for prediction: (n_pixels, n_bands)
            h, w = features.shape[1], features.shape[2]
            X_pred = features.reshape(features.shape[0], -1).T  # (n_pixels, n_bands)
            
            # Handle NaN values for prediction (set to 0)
            X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)
            
            print(f"Predicting on {X_pred.shape[0]} pixels with {X_pred.shape[1]} features...")
            
            if args.model == 'rf':
                predictions = model.predict(X_pred)
            else:  # MLP
                model.eval()
                with torch.no_grad():
                    X_pred_tensor = torch.FloatTensor(X_pred)
                    X_pred_normalized = (X_pred_tensor - model.scaler_mean) / model.scaler_std
                    
                    predictions = []
                    batch_size = min(args.batch_size, 1024)  # Limit batch size for memory
                    
                    for i in tqdm(range(0, len(X_pred), batch_size), desc="Predicting batches"):
                        batch = X_pred_normalized[i:i + batch_size]
                        if torch.cuda.is_available():
                            batch = batch.cuda()
                        pred = model(batch)
                        predictions.extend(pred.cpu().numpy())
                    predictions = np.array(predictions)
            
            # Reshape back to spatial dimensions
            predictions = predictions.reshape(h, w)
            print(f"Generated prediction map with shape: {predictions.shape}")
            print(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
            
        else:  # U-Net models
            model.eval()
            with torch.no_grad():
                # Add batch dimension
                if args.model == '2d_unet':
                    # features: (bands, h, w) -> (1, bands, h, w)
                    input_tensor = torch.FloatTensor(features).unsqueeze(0)
                else:  # 3d_unet
                    # Reshape temporal data using same logic as train_3d_unet
                    n_bands = features.shape[0]
                    h, w = features.shape[1], features.shape[2]
                    
                    # Expected structure: 180 temporal + 14 non-temporal = 194 total feature bands
                    n_temporal_expected = 180  # 15 bands × 12 months
                    
                    if n_bands >= n_temporal_expected:
                        # Separate temporal and non-temporal bands
                        temporal_features = features[:n_temporal_expected]  # First 180 bands
                        nontemporal_features = features[n_temporal_expected:] if n_bands > n_temporal_expected else np.empty((0, h, w))
                        
                        # Reshape temporal data: (180, h, w) -> (15, 12, h, w)
                        n_channels_per_month = 15  # S1(2) + S2(11) + ALOS2(2) = 15
                        n_months = 12
                        
                        temporal_features_reshaped = temporal_features.reshape(n_channels_per_month, n_months, h, w)
                        
                        # Handle missing values in temporal data
                        temporal_features_reshaped = np.nan_to_num(temporal_features_reshaped, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # Combine temporal and non-temporal features
                        if len(nontemporal_features) > 0:
                            # Repeat non-temporal features across time dimension
                            nontemporal_expanded = np.tile(nontemporal_features[:, np.newaxis, :, :], (1, n_months, 1, 1))
                            # Combine temporal and non-temporal
                            combined_features = np.concatenate([temporal_features_reshaped, nontemporal_expanded], axis=0)
                        else:
                            combined_features = temporal_features_reshaped
                        
                        # Check if we need to apply the same fallback logic as in training
                        try:
                            # Test with small input to see if 3D works
                            test_input = torch.randn(1, combined_features.shape[0], 12, 32, 32)
                            # Try to load the model and see if it expects 3D or 2D input
                            with torch.no_grad():
                                if hasattr(model, 'encoder1'):  # 2D U-Net (fallback was used)
                                    # Use temporal averaging for prediction too
                                    averaged_features = np.mean(combined_features, axis=1)  # Average across time
                                    input_tensor = torch.FloatTensor(averaged_features).unsqueeze(0)
                                    print(f"Using temporal averaging for prediction: {averaged_features.shape}")
                                else:  # 3D U-Net
                                    input_tensor = torch.FloatTensor(combined_features).unsqueeze(0)
                        except:
                            # Default to temporal averaging if we can't determine
                            averaged_features = np.mean(combined_features, axis=1)  # Average across time
                            input_tensor = torch.FloatTensor(averaged_features).unsqueeze(0)
                            print(f"Using temporal averaging for prediction (fallback): {averaged_features.shape}")
                    else:
                        raise ValueError(f"Insufficient bands for temporal processing: got {n_bands}, expected at least {n_temporal_expected}")
                
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                    model = model.cuda()
                
                predictions = model(input_tensor)
                predictions = predictions.squeeze().cpu().numpy()
        
        # Save prediction
        if args.prediction_output is None:
            patch_name = os.path.splitext(os.path.basename(args.patch_path))[0]
            pred_filename = f"prediction_{args.model}_{patch_name}.tif"
            prediction_path = os.path.join(args.output_dir, pred_filename)
        else:
            prediction_path = args.prediction_output
        
        # Save with same georeference as input patch
        with rasterio.open(args.patch_path) as src:
            profile = src.profile.copy()
            profile.update(count=1, dtype='float32')
            
            with rasterio.open(prediction_path, 'w', **profile) as dst:
                dst.write(predictions.astype('float32'), 1)
        
        print(f"Saved prediction to: {prediction_path}")
    
    print("\\nTraining completed successfully!")

if __name__ == "__main__":
    main()