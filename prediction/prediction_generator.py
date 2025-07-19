import numpy as np
import torch
from scipy.ndimage import zoom
from typing import Tuple

from data.multi_patch import PatchInfo
from data.patch_loader import load_patch_data

def generate_patch_prediction(model, patch_info: PatchInfo, model_type: str, is_temporal: bool, 
                              supervision_mode: str = "gedi", band_selection: str = "all") -> np.ndarray:
    """Generate prediction for a single patch using trained model."""
    # Load patch data with specified band selection
    features, _, _ = load_patch_data(patch_info.file_path, 
                                   supervision_mode=supervision_mode, 
                                   band_selection=band_selection, 
                                   normalize_bands=True)
    
    # Store original dimensions for reshaping output
    original_height, original_width = features.shape[-2], features.shape[-1]
    
    if model_type in ['rf', 'mlp']:
        # Prepare full feature data for traditional models
        full_features = features.reshape(features.shape[0], -1).T  # (n_pixels, n_bands)
        
        # Remove any NaN/inf values
        valid_mask = np.all(np.isfinite(full_features), axis=1)
        
        if model_type == 'rf':
            predictions = np.full(valid_mask.shape[0], np.nan)
            predictions[valid_mask] = model.predict(full_features[valid_mask])
        else:  # MLP
            if torch.cuda.is_available():
                model.eval()
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(full_features[valid_mask])
                    if torch.cuda.is_available():
                        input_tensor = input_tensor.cuda()
                        model = model.cuda()
                    pred_tensor = model(input_tensor)
                    pred_values = pred_tensor.cpu().numpy().flatten()
                    
                predictions = np.full(valid_mask.shape[0], np.nan)
                predictions[valid_mask] = pred_values
            else:
                raise ImportError("PyTorch is required for MLP prediction")
        
        # Reshape to original dimensions
        predictions = predictions.reshape(features.shape[1], features.shape[2])
        
    else:  # U-Net models
        # Resize features to 256x256 if needed for U-Net models
        if features.shape[-2] != 256 or features.shape[-1] != 256:
            print(f"Resizing patch from {features.shape[-2]}x{features.shape[-1]} to 256x256")
            
            if len(features.shape) == 3:  # (bands, height, width)
                scale_h = 256 / features.shape[1]
                scale_w = 256 / features.shape[2]
                resized_features = np.zeros((features.shape[0], 256, 256), dtype=features.dtype)
                for i in range(features.shape[0]):
                    resized_features[i] = zoom(features[i], (scale_h, scale_w), order=1)
                features = resized_features
            elif len(features.shape) == 4:  # (bands, time, height, width)
                scale_h = 256 / features.shape[2]
                scale_w = 256 / features.shape[3]
                resized_features = np.zeros((features.shape[0], features.shape[1], 256, 256), dtype=features.dtype)
                for i in range(features.shape[0]):
                    for j in range(features.shape[1]):
                        resized_features[i, j] = zoom(features[i, j], (scale_h, scale_w), order=1)
                features = resized_features
        
        if model_type == '2d_unet':
            # Non-temporal mode
            if len(features.shape) == 4:  # (bands, time, height, width)
                combined_features = features.reshape(-1, features.shape[2], features.shape[3])
            else:  # (bands, height, width)
                combined_features = features
                
            input_tensor = torch.FloatTensor(combined_features).unsqueeze(0)
            
        
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            model = model.cuda()
        
        model.eval()
        with torch.no_grad():
            predictions = model(input_tensor)
            predictions = predictions.squeeze().cpu().numpy()
            
        # Resize predictions back to original dimensions if needed
        if predictions.shape != (original_height, original_width):
            scale_h = original_height / predictions.shape[0]
            scale_w = original_width / predictions.shape[1]
            predictions = zoom(predictions, (scale_h, scale_w), order=1)
    
    return predictions
