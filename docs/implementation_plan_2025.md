# Implementation Plan: Converting CHM_Image to Image-Based Approach with 3D U-Net

## 1. Data Collection and Preprocessing Updates

### 1.1 Data Sources (Maintaining Current Inputs) 
```python
# Keep existing data sources from chm_main.py:
- Sentinel-1 (VV & VH polarizations, median composite)
- Sentinel-2 (10 optical bands, 12-month time series)
- ALOS-2 SAR data
- DEM data
- Canopy height data
- GEDI L2A data

* data collected between start date and end data (e.g. 2022-01-01 to 2022-12-31, usually whole year)
* all data should be resampled to scale (defaults 10m, but try 30m for test)
```

### 1.2 Data Processing Updates
```python
# New file: data/image_patches.py
def create_3d_patches(region, patch_size=2560, overlap=0.0, scale=10):
    """
    Create 3D image patches for training
    - patch_size: 2.56km (2560m) or 256 pixels at 10m resolution
    - overlap: no overlap between patches for default
    - Stack all input bands as channels
    - Include 12-month median time series of Sentinel-2
    - scale: resolution in meters (default=10m, can be 30m for testing)
    """
    pass

def prepare_training_patches(merged_data, gedi_data, patch_size=2560, scale=10):
    """
    Prepare 3D training patches with GEDI labels
    - Convert GEDI points to patch-level labels
    - Stack all input bands as channels
    - Handle patch-level data augmentation
    - Include temporal information
    - Scaling to 10m resolution raster from 25m shot beam as defaults
    """
    pass

# New file: data/large_area.py
def create_patch_grid(area_bounds, patch_size=2560, overlap=0, scale=10):
    """
    Create a grid of patches for large area mapping, handling non-divisible areas
    
    Args:
        area_bounds: (min_x, min_y, max_x, max_y) in meters
        patch_size: Size of each patch in meters (default=2560m)
        overlap: Overlap between patches in meters (default=0)
        scale: Resolution in meters (default=10m, can be 30m for testing)
    
    Returns:
        List of patch coordinates (x, y, width, height)
        
    Notes:
        - If area is not perfectly divisible by patch_size, patches will extend slightly beyond bounds
        - For 10m resolution: patch_size=2560m (256 pixels)
        - For 30m resolution: patch_size=7680m (256 pixels)
        - Extruded areas will be masked out in final output
    """
    # Calculate number of patches needed
    width = area_bounds[2] - area_bounds[0]
    height = area_bounds[3] - area_bounds[1]
    
    # Calculate number of patches (rounding up to ensure coverage)
    n_patches_x = int(np.ceil(width / patch_size))
    n_patches_y = int(np.ceil(height / patch_size))
    
    # Calculate actual patch size needed to cover area
    actual_patch_size_x = width / n_patches_x
    actual_patch_size_y = height / n_patches_y
    
    patches = []
    for i in range(n_patches_x):
        for j in range(n_patches_y):
            x = area_bounds[0] + i * actual_patch_size_x
            y = area_bounds[1] + j * actual_patch_size_y
            patches.append({
                'x': x,
                'y': y,
                'width': actual_patch_size_x,
                'height': actual_patch_size_y,
                'is_extruded': (
                    x + actual_patch_size_x > area_bounds[2] or
                    y + actual_patch_size_y > area_bounds[3]
                )
            })
    
    return patches

def collect_area_patches(area_bounds, data_sources, patch_size=2560):
    """
    Collect and process patches for a large area
    
    Args:
        area_bounds: (min_x, min_y, max_x, max_y) in meters
        data_sources: Dictionary of data sources (S1, S2, etc.)
        patch_size: Size of each patch in meters
    
    Returns:
        Dictionary of processed patches
    """
    pass

def merge_patch_predictions(predictions, area_bounds, patch_size=2560, scale=10):
    """
    Merge predictions from individual patches into a single map
    
    Args:
        predictions: Dictionary of patch predictions
        area_bounds: Original area bounds
        patch_size: Size of each patch in meters
        scale: Resolution in meters (default=10m, can be 30m for testing)
    
    Returns:
        Merged prediction map with masked out extruded areas
    """
    pass

# New file: evaluation/large_area_report.py
def generate_large_area_report(predictions, area_bounds, data_sources, output_dir):
    """
    Generate comprehensive evaluation report for large area predictions
    
    Args:
        predictions: Dictionary of patch predictions
        area_bounds: (min_x, min_y, max_x, max_y) in meters
        data_sources: Dictionary of data sources used
        output_dir: Directory to save report
    """
    pass

def calculate_area(bounds):
    """Calculate area in hectares from bounds."""
    pass

### 1.3 Scale and Resolution Handling
```python
# New file: config/resolution_config.py
RESOLUTION_CONFIG = {
    'default_scale': 10,  # Default resolution in meters
    'possible_another_scale': 30,     # alternative resolution in meters
    'patch_sizes': {
        10: 2560,  # 256 pixels × 10m = 2560m
        30: 7680   # 256 pixels × 30m = 7680m
    },
    'pixel_counts': {
        10: 256,  # 2560m ÷ 10m = 256 pixels
        30: 256   # 7680m ÷ 30m = 256 pixels
    }
}

def get_patch_size(scale):
    """
    Get appropriate patch size for given resolution
    
    Args:
        scale: Resolution in meters (10 or 30)
    
    Returns:
        Patch size in meters that maintains 256x256 pixels
        
    Example:
        For 10m resolution: 256 pixels × 10m = 2560m
        For 30m resolution: 256 pixels × 30m = 7680m
    """
    return RESOLUTION_CONFIG['patch_sizes'].get(scale, 2560)
```

### 1.4 Handling Non-Divisible Areas
When processing areas that don't perfectly divide by the patch size:

1. **Patch Grid Creation**:
   - Calculate number of patches needed by rounding up
   - Adjust patch sizes to ensure complete coverage
   - Mark patches that extend beyond original bounds
   - Maintain consistent pixel count (256x256) regardless of resolution

2. **Processing Strategy**:
   - Process all patches including extruded ones
   - Keep track of which patches are extruded
   - Apply masking to remove predictions in extruded areas
   - Ensure consistent pixel dimensions across resolutions

3. **Resolution Handling**:
   - Support both 10m and 30m resolutions
   - Adjust patch sizes to maintain 256x256 pixels:
     * 10m resolution: 2560m × 2560m patches
     * 30m resolution: 7680m × 7680m patches
   - Ensure consistent processing across resolutions

4. **Output Generation**:
   - Merge predictions from all patches
   - Apply masking to remove extruded areas
   - Ensure final output matches original area bounds
   - Maintain consistent pixel dimensions in output
```

### 1.5 Input Normalization Strategy

All input bands should be normalized before being used for model training. The normalization strategies for each data source are as follows:

```python
# New file: data/normalization.py

def normalize_sentinel1(band):
    # Shift by +25, then divide by 25
    return (band + 25) / 25

def normalize_sentinel2(band):
    # Divide by 10,000
    return band / 10000

def normalize_srtm_elevation(elev_m):
    # Divide by 2,000
    return elev_m / 2000

def normalize_srtm_slope(slope_deg):
    # Divide by 50
    return slope_deg / 50

def normalize_srtm_aspect(aspect_deg):
    # Shift by -180, then divide by 180
    return (aspect_deg - 180) / 180

def normalize_alos2_dn(dn):
    # Convert DN to gamma_naught_dB, then use as input
    # gamma_naught_dB = 10 * np.log10(DN**2) - 83.0
    import numpy as np
    return 10 * np.log10(dn ** 2) - 83.0

def normalize_canopy_height(height_m):
    # Divide by 50
    return height_m / 50

# NDVI does not require normalization
def normalize_ndvi(ndvi):
    return ndvi
```

**Implementation Steps:**
1. Apply the above normalization functions to each corresponding band/variable during data preprocessing.
2. Ensure that the same normalization is used for both training and prediction.
3. Document the normalization in the codebase and in the data pipeline documentation.

## 2. Model Architecture Updates

### 2.1 3D U-Net Implementation
```python
# New file: models/3d_unet.py
import torch
import torch.nn as nn

num_inputBand = ? # 12 months * 10 S2 bands + 4 S1 bands + AlOS2 + DEM + canopy height data but sometimes change
class Height3DUNet(nn.Module):
    def __init__(self, in_channels=num_inputBand):  
        super().__init__()
        # 3D convolutions for processing temporal data
        self.encoder = nn.ModuleList([
            # Initial 3D conv block
            nn.Conv3d(in_channels, 64, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # Downsampling blocks
            nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(128, 256, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(256, 512, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        ])
        
        # U-Net style decoder with 3D upsampling
        self.decoder = nn.ModuleList([
            # Upsampling blocks
            UpConv3DBlock(512, 256),
            UpConv3DBlock(256, 128),
            UpConv3DBlock(128, 64),
            UpConv3DBlock(64, 32)
        ])
        
        # Final prediction head (2D output)
        self.head = nn.Conv2d(32, 1, kernel_size=1)
    
    def forward(self, x):
        # x shape: [batch, channels, time=12, height, width]
        # Encoder path
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        
        # Decoder path
        for i, decoder_block in enumerate(self.decoder):
            x = decoder_block(x, features[-(i+2)])
        
        # Final prediction (2D output)
        return self.head(x.squeeze(2))  # Remove time dimension
```

### 2.2 Modified Huber Loss
```python
# Update dl_models.py
def modified_huber_loss(pred, target, mask=None, delta=1.0, shift_radius=1):
    """
    Modified Huber loss for 3D patches
    - Handles spatial shift in GEDI data
    - Applies loss only on valid pixels
    - Includes patch-level weighting
    - Supports configurable shift radius (default=1)
    """
    def huber_loss(x, y, delta=1.0):
        diff = x - y
        abs_diff = diff.abs()
        quadratic = torch.min(abs_diff, torch.tensor(delta))
        linear = abs_diff - quadratic
        return torch.mean(0.5 * quadratic.pow(2) + delta * linear)
    
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
        shifted_target = torch.roll(target, shifts=(dx, dy), dims=(2, 3))
        
        # Compute loss
        if mask is not None:
            loss = huber_loss(pred * mask, shifted_target * mask)
        else:
            loss = huber_loss(pred, shifted_target)
        
        best_loss = min(best_loss, loss)
    
    return best_loss
```

## 3. Training Pipeline Updates

### 3.1 Data Loading
- Create 3D patches from all input bands
- Implement patch-level augmentation
- Add patch-level validation
- Handle temporal data

### 3.2 Training Configuration
```python
# New file: config/training_config.py
TRAINING_CONFIG = {
    'patch_size': 2560,  # 2.56km patches
    'batch_size': 8,     # Smaller batch size due to 3D processing
    'learning_rate': 1e-3,
    'weight_decay': 1e-3,
    'warmup_steps': 0.1,  # 10% of total steps
    'total_steps': 100,
    'gradient_clip_val': 1.0,
    'huber_delta': 1.0,
    'temporal_smoothing': 5  # Smoothing factor for temporal predictions
}
```

## 4. Implementation Steps

1. **Data Processing (Week 1)**
   - Modify `chm_main.py` to support 3D patch creation
   - Implement 3D patch management
   - Update data loading pipeline
   - Add temporal data handling
   - Implement large area patch collection
   - Add patch merging functionality
   # try small region and visualize and check statics and whether data is correctly extracted.

2. **Model Development (Week 2)**
   - Implement 3D U-Net architecture
   - Add modified Huber loss
   - Implement data augmentation
   - Add temporal smoothing
   - Add patch merging functionality
   - Add large area evaluation report generation
   - try one patch with 5 iteration with 2 epochs. 

3. **Training Pipeline (Week 3)**
   - Set up 3D patch-based training
   - Implement validation on patch level
   - Add model checkpointing
   - Add temporal prediction handling
   - Test large area processing
   - Generate evaluation reports

4. **Evaluation and Testing (Week 4)**
   - Implement patch-level evaluation
   - Add visualization tools
   - Compare with pixel-based results
   - Evaluate temporal predictions
   - Test large area mapping
   - Generate comprehensive reports

## 5. Required Dependencies

Update `requirements.txt`:
```
torch>=1.9.0
torchvision>=0.10.0
earthengine-api>=0.1.323
scipy>=1.7.0
numpy>=1.19.0
pandas>=1.3.0
rasterio>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
albumentations>=1.0.0  # For image augmentation
```

## 6. Testing Strategy

1. **Unit Tests**
   - Test 3D patch creation and management
   - Test 3D U-Net architecture
   - Test modified Huber loss
   - Test temporal smoothing

2. **Integration Tests**
   - Test end-to-end training pipeline
   - Test inference pipeline
   - Test data augmentation
   - Test temporal predictions

3. **Validation Tests**
   - Compare with GEDI validation data
   - Test on different patch sizes
   - Evaluate model performance
   - Evaluate temporal consistency

## 7. Documentation Updates

1. Update README.md with new methodology
2. Add API documentation for new functions
3. Create usage examples for 3D patch-based processing
4. Document training process
5. Document temporal prediction handling

## 8. Performance Metrics

Track the following metrics:
- Patch-level MAE
- Patch-level RMSE
- Overall height distribution accuracy
- Computational efficiency
- Temporal prediction accuracy
- Tall tree (>30m) prediction accuracy

## Next Steps

1. Review and approve implementation plan
2. Set up development environment
3. Begin with data processing pipeline
4. Implement 3D U-Net architecture
5. Set up training pipeline
6. Run initial tests
7. Iterate based on results

## Key Changes from Current Implementation

1. **Data Processing**:
   - Convert pixel-based to 3D patch-based processing
   - Stack all input bands as channels in 3D patches
   - Maintain same input data sources
   - Normalize each inputs bands.
   - Add patch-level data augmentation (only rotation)
   - Add temporal data handling
   - Add large area patch collection and merging
   - Add comprehensive evaluation reporting

2. **Model Architecture**:
   - Implement 3D U-Net for processing stacked bands
   - Use 3D convolutions for spatial feature extraction
   - Add modified Huber loss for GEDI data alignment
   - Add temporal prediction handling
   - Support large area processing
   - Add evaluation report generation

3. **Training Process**:
   - Randomly select training patches and validation patches.
   - Update batch processing for 3D patches
   - Implement patch-level loss functions
   - Add patch-level validation
   - Add temporal smoothing
   - Add patch merging for large areas
   - Add automated report generation

4. **Evaluation**:
   - Add patch-level evaluation metrics
   - Implement visualization tools
   - Compare with current pixel-based results
   - Evaluate temporal predictions
   - Test large area mapping accuracy
   - Generate comprehensive PDF reports 