"""
Caching utilities for multi-patch training data.
"""

import os
import numpy as np
from typing import Tuple, Optional, List
from .multi_patch import PatchInfo, load_multi_patch_gedi_data, load_multi_patch_reference_data, load_multi_patch_reference_data_parallel


def create_cache_filename(args, data_type: str = 'reference') -> str:
    """
    Create descriptive cache filename based on dataset parameters.
    
    Args:
        args: Training arguments
        data_type: 'reference' or 'gedi'
        
    Returns:
        Cache filename with descriptive parameters
    """
    if data_type == 'reference':
        ref_filename = os.path.basename(args.reference_height_path).replace('.tif', '')
        patch_pattern_clean = args.patch_pattern.replace('*', '').replace('.tif', '')
        cache_name = f"cached_{ref_filename}_{patch_pattern_clean}_ref{args.min_reference_samples}_{args.supervision_mode}_data.npz"
    else:  # GEDI
        patch_pattern_clean = args.patch_pattern.replace('*', '').replace('.tif', '')
        cache_name = f"cached_{patch_pattern_clean}_gedi{args.min_gedi_samples}_data.npz"
    
    return cache_name


def load_or_create_reference_data(patches: List[PatchInfo], args) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load reference data from cache or create new data and cache it.
    
    Args:
        patches: List of patch metadata
        args: Training arguments
        
    Returns:
        Tuple of (combined_features, combined_targets)
    """
    cache_name = create_cache_filename(args, 'reference')
    cache_file = os.path.join(args.output_dir, cache_name)
    
    if os.path.exists(cache_file):
        print(f"💾 Loading cached reference data: {cache_name}")
        cached_data = np.load(cache_file)
        combined_features = cached_data['features']
        combined_targets = cached_data['targets']
        print(f"✅ Loaded cached data: Features {combined_features.shape}, Targets {combined_targets.shape}")
        return combined_features, combined_targets
    else:
        print("🔄 No cache found, loading reference data with parallel processing (~15 minutes)...")
        print(f"📝 Cache will be saved as: {cache_name}")
        
        # Apply reference filtering only in training mode
        min_reference_samples = args.min_reference_samples if args.mode == 'train' else 0
        
        # Try parallel loading first, fall back to sequential if it fails
        try:
            print("⚡ Attempting parallel loading with 8 workers...")
            combined_features, combined_targets = load_multi_patch_reference_data_parallel(
                patches, 
                reference_tif_path=args.reference_height_path,
                min_reference_samples=min_reference_samples,
                n_workers=8  # Conservative number to avoid OOM
            )
        except Exception as e:
            print(f"⚠️  Parallel loading failed ({e}), falling back to sequential loading...")
            combined_features, combined_targets = load_multi_patch_reference_data(
                patches, 
                reference_tif_path=args.reference_height_path,
                min_reference_samples=min_reference_samples
            )
        
        # Save to cache for future use
        print(f"💾 Saving reference data to cache: {cache_name}")
        np.savez_compressed(cache_file, features=combined_features, targets=combined_targets)
        print(f"✅ Cached data saved for future use ({cache_file})")
        return combined_features, combined_targets


def load_or_create_gedi_data(patches: List[PatchInfo], args) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load GEDI data from cache or create new data and cache it.
    
    Args:
        patches: List of patch metadata
        args: Training arguments
        
    Returns:
        Tuple of (combined_features, combined_targets)
    """
    cache_name = create_cache_filename(args, 'gedi')
    cache_file = os.path.join(args.output_dir, cache_name)
    
    if os.path.exists(cache_file):
        print(f"💾 Loading cached GEDI data: {cache_name}")
        cached_data = np.load(cache_file)
        combined_features = cached_data['features']
        combined_targets = cached_data['targets']
        print(f"✅ Loaded cached data: Features {combined_features.shape}, Targets {combined_targets.shape}")
        return combined_features, combined_targets
    else:
        print("🔄 No cache found, loading GEDI data...")
        print(f"📝 Cache will be saved as: {cache_name}")
        
        # Apply GEDI filtering only in training mode
        min_gedi_samples = args.min_gedi_samples if args.mode == 'train' else 0
        combined_features, combined_targets = load_multi_patch_gedi_data(
            patches, target_band='rh', min_gedi_samples=min_gedi_samples
        )
        
        # Save to cache for future use
        print(f"💾 Saving GEDI data to cache: {cache_name}")
        np.savez_compressed(cache_file, features=combined_features, targets=combined_targets)
        print(f"✅ Cached data saved for future use ({cache_file})")
        return combined_features, combined_targets