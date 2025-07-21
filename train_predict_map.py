import numpy as np
import pandas as pd
try:
    import torch
    import torch.optim as optim
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. U-Net will not be functional.")

import rasterio
import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import warnings
import argparse
from tqdm import tqdm
import glob
import json
import joblib
import time
warnings.filterwarnings('ignore')

# Import multi-patch functionality
from data.multi_patch import (
    PatchInfo, PatchRegistry, PredictionMerger,
    load_multi_patch_gedi_data, load_multi_patch_reference_data,
    generate_multi_patch_summary, count_gedi_samples_per_patch
)

# Import caching utilities
from data.cache_utils import (
    load_or_create_reference_data, load_or_create_gedi_data
)

from data.patch_loader import load_patch_data

# Import shift-aware training functionality
try:
    from models.trainers.shift_aware_trainer import ShiftAwareTrainer, auto_find_patches
    SHIFT_AWARE_AVAILABLE = True
except ImportError:
    SHIFT_AWARE_AVAILABLE = False
    print("Warning: Shift-aware trainer not available")

# Import mosaic utilities
try:
    from utils.mosaic_utils import create_comprehensive_mosaic, find_all_patches
    MOSAIC_UTILS_AVAILABLE = True
except ImportError:
    MOSAIC_UTILS_AVAILABLE = False
    print("Warning: Mosaic utilities not available")

# Import enhanced spatial merger
try:
    from utils.spatial_utils import EnhancedSpatialMerger
    USE_ENHANCED_MERGER = True
except ImportError:
    USE_ENHANCED_MERGER = False
    print("Warning: Enhanced spatial merger not available, using default merger")

from models.height_2d_unet import Height2DUNet
from models.trainers.enhanced_unet_trainer import EnhancedUNetTrainer

# Import refactored modules
from utils.data_processing_utils import apply_band_normalization, extract_sparse_gedi_pixels, detect_temporal_mode, separate_temporal_nontemporal_bands
from data.data_loader import load_patches_from_directory, load_training_data, load_prediction_data
from models.trainers.unet_trainer import train_2d_unet, train_multi_patch_unet_reference
from models.trainers.traditional_trainer import train_model
from prediction.prediction_saver import save_predictions
from prediction.prediction_generator import generate_patch_prediction
from utils.reporting_utils import save_metrics_and_importance, save_training_metrics

def create_2d_unet(in_channels: int, n_classes: int = 1, base_channels: int = 64):
    """Create 2D U-Net model."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for 2D U-Net")
    return Height2DUNet(in_channels, n_classes, base_channels)


def train_multi_patch(args):
    """Train model on multiple patches with unified training and optional prediction merging."""
    print(f"🚀 Starting multi-patch training with {args.model.upper()}")
    
    # Initialize patch registry and discover patches
    patch_registry = PatchRegistry()
    patches = patch_registry.discover_patches(args.patch_dir, args.patch_pattern)
    
    if not patches:
        raise ValueError(f"No patches found in {args.patch_dir} matching pattern {args.patch_pattern}")
    
    print(f"📊 Discovered {len(patches)} patches")
    
    # Handle shift-aware U-Net training early (bypass unified data loading)
    if args.model == 'shift_aware_unet':
        if not SHIFT_AWARE_AVAILABLE:
            raise ImportError("Shift-aware trainer not available. Please ensure models/trainers/shift_aware_trainer.py is accessible.")
        
        print("🔄 Using advanced shift-aware U-Net training...")
        
        # Use command-line specified patch discovery and splitting
        registry = PatchRegistry()
        patches = registry.discover_patches(args.patch_dir, args.patch_pattern)
        
        if len(patches) < 4:
            raise ValueError(f"Insufficient patches found: {len(patches)} (minimum 4 required)")
        
        # Split patches into train/validation using PatchInfo objects
        patch_files = [patch.file_path for patch in patches]
        from sklearn.model_selection import train_test_split
        train_patches, val_patches = train_test_split(
            patch_files, 
            test_size=0.3, 
            random_state=42
        )
        
        # Initialize trainer with optimal settings from experiments
        trainer = ShiftAwareTrainer(
            shift_radius=args.shift_radius,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            band_selection=getattr(args, 'band_selection', 'all'),
            pretrained_model_path=getattr(args, 'pretrained_model_path', None)
        )
        
        # Train model
        training_history = trainer.train(
            train_patches=train_patches,
            val_patches=val_patches,
            epochs=args.epochs,
            output_dir=args.output_dir
        )
        
        model_path = training_history['model_path']
        print(f"💾 Shift-aware U-Net saved to: {model_path}")
        
        # Create comprehensive mosaic if requested
        if args.generate_prediction and MOSAIC_UTILS_AVAILABLE:
            print("🗺️  Creating comprehensive prediction mosaic...")
            mosaic_name = f"comprehensive_mosaic_{args.model}.tif"
            mosaic_metadata = create_comprehensive_mosaic(model_path, mosaic_name)
            print(f"📁 Comprehensive mosaic: {mosaic_name}")
        
        return  # Exit early for shift-aware training
    
    # Validate patch consistency
    is_consistent = patch_registry.validate_consistency()
    if not is_consistent:
        print("⚠️  Warning: Inconsistencies detected in patches. Proceeding with caution...")
    
    # Generate and save patch summary
    summary_df = generate_multi_patch_summary(patches)
    summary_path = os.path.join(args.output_dir, 'patch_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"💾 Saved patch summary to: {summary_path}")
    
    # Get summary statistics
    summary_stats = patch_registry.get_patch_summary()
    print(f"📈 Dataset summary:")
    print(f"  - Total patches: {summary_stats['total_patches']}")
    print(f"  - Temporal patches: {summary_stats['temporal_patches']}")
    print(f"  - Non-temporal patches: {summary_stats['non_temporal_patches']}")
    print(f"  - Total area: {summary_stats['total_area_km2']:.1f} km²")
    print(f"  - Reference CRS: {summary_stats['reference_crs']}")
    print(f"  - Reference resolution: {summary_stats['reference_resolution']}m")
    print(f"  - Reference bands: {summary_stats['reference_bands']}")
    
    # Detect temporal mode from first patch
    reference_patch = patches[0]
    is_temporal = reference_patch.temporal_mode
    print(f"🕐 Temporal mode detected: {is_temporal}")
    
    # Validate model-data compatibility
    if args.model == '2d_unet' and is_temporal:
        raise ValueError("2D U-Net cannot be used with temporal data. Use '3d_unet' or use non-temporal patches.")
    
    # Choose training strategy based on model type and available data
    if args.model == '2d_unet' and hasattr(args, 'supervision_mode') and args.supervision_mode in ['reference', 'reference_only']:
        
        # Check if enhanced patches with pre-processed reference bands are available
        enhanced_patch_dir = os.path.join(os.path.dirname(args.patch_dir), 'enhanced_patches')
        enhanced_patches = glob.glob(os.path.join(enhanced_patch_dir, "ref_*05LE4*.tif"))
        
        if len(enhanced_patches) > 0:
            # Use ultra-fast training with enhanced patches
            print("⚡ Using ULTRA-FAST training with enhanced patches")
            print(f"📁 Enhanced patches directory: {enhanced_patch_dir}")
            print(f"🔍 Found {len(enhanced_patches)} enhanced patches")
            print("✅ No runtime TIF loading needed - reference data is pre-processed!")
            
            from fast_training_with_enhanced_patches import train_ultra_fast_unet
            model, train_metrics = train_ultra_fast_unet(
                patch_dir=enhanced_patch_dir,
                output_dir=args.output_dir,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                base_channels=args.base_channels,
                validation_split=args.validation_split
            )
            
        else:
            # Fallback to standard batch training with runtime TIF loading
            print("🚀 Using standard batch training (runtime TIF loading)")
            print(f"🏔️  Using reference height supervision: {args.supervision_mode}")
            print(f"📁 Reference TIF: {args.reference_height_path}")
            print("💡 Tip: Use preprocess_reference_bands.py to create enhanced patches for 10x+ speedup")
            
            from fast_batch_training import train_fast_batch_unet
            model, train_metrics = train_fast_batch_unet(
                patches, args,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                base_channels=args.base_channels,
                validation_split=args.validation_split
            )
        
        # Save the model
        model_path = os.path.join(args.output_dir, 'multi_patch_2d_unet_model.pth')
        if TORCH_AVAILABLE:
            torch.save(model.state_dict(), model_path)
            print(f"💾 Saved 2D U-Net model to: {model_path}")
        
        # Save training metrics
        metrics_path = os.path.join(args.output_dir, 'multi_patch_training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(train_metrics, f, indent=2)
        print(f"📊 Training metrics saved to: {metrics_path}")
            
        # Early return - skip the rest of the function
        return
    
    # Load multi-patch training data for other models
    print("📖 Loading training data from all patches...")
    
    # Choose data loading strategy based on supervision mode
    if hasattr(args, 'supervision_mode') and args.supervision_mode in ['reference', 'reference_only']:
        if not hasattr(args, 'reference_height_path') or not args.reference_height_path:
            raise ValueError("Reference height path must be specified for reference supervision")
        
        print(f"🏔️  Using reference height supervision: {args.supervision_mode}")
        print(f"📁 Reference TIF: {args.reference_height_path}")
        
        # Use cached reference data loading (avoids 20+ minute loading time)
        combined_features, combined_targets = load_or_create_reference_data(patches, args)
        
    else:
        # Default GEDI supervision
        print("🛰️  Using GEDI supervision (sparse)")
        
        # Use cached GEDI data loading
        combined_features, combined_targets = load_or_create_gedi_data(patches, args)
    
    print(f"🎯 Combined training dataset:")
    print(f"  - Features shape: {combined_features.shape}")
    print(f"  - Targets shape: {combined_targets.shape}")
    print(f"  - Target range: {combined_targets.min():.1f} - {combined_targets.max():.1f}m")
    
    # Train unified model
    print(f"🏋️ Training unified {args.model.upper()} model...")
    
    if args.model in ['rf', 'mlp']:
        # Traditional models can train directly on combined features
        feature_names = [f'band_{i+1}' for i in range(combined_features.shape[1])]
        
        model, train_metrics, importance_data = train_model(
            combined_features, combined_targets,
            model_type=args.model,
            batch_size=args.batch_size,
            test_size=args.test_size,
            feature_names=feature_names
        )
        
        # Save model
        if args.model == 'rf':
            model_path = os.path.join(args.output_dir, 'multi_patch_rf_model.pkl')
            joblib.dump(model, model_path)
        else:  # MLP
            model_path = os.path.join(args.output_dir, 'multi_patch_mlp_model.pth')
            if TORCH_AVAILABLE:
                torch.save(model.state_dict(), model_path)
        
        print(f"💾 Saved {args.model.upper()} model to: {model_path}")
        
    else:
        # U-Net multi-patch training with improved approach
        print("🚀 Using improved U-Net multi-patch training")
        
        # For U-Net models, need to handle reference vs GEDI supervision differently
        print(f"📊 Training on {len(patches)} filtered patches")
        if hasattr(args, 'supervision_mode') and args.supervision_mode in ['reference', 'reference_only']:
            print(f"📊 Total reference samples: {len(combined_targets)}")
            supervision_type = "reference"
        else:
            print(f"📊 Total GEDI samples: {len(combined_targets)}")
            supervision_type = "gedi"
        print(f"📊 Using validation split: {args.validation_split}")
        
        if args.model == '2d_unet':
            if supervision_type == "reference":
                # Use image-based U-Net training with proper spatial data
                print("🚀 Using image-based U-Net training (proper spatial approach)")
                
                # Get input channels from first patch to determine model architecture
                reference_patch = patches[0]
                sample_features, _, _ = load_patch_data(
                    reference_patch.file_path, 
                    supervision_mode=args.supervision_mode,
                    band_selection=getattr(args, 'band_selection', 'all'),
                    normalize_bands=True
                )
                in_channels = sample_features.shape[0]
                print(f"📊 Detected {in_channels} input channels from {getattr(args, 'band_selection', 'all')} band selection")
                
                model, train_metrics = train_multi_patch_unet_reference(
                    patches, args,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    base_channels=args.base_channels,
                    validation_split=args.validation_split
                )
            else:
                # Use first patch for GEDI supervision (existing method)
                reference_patch = patches[0]
                first_patch_features, first_patch_target, band_info = load_patch_data(
                    reference_patch.file_path, 
                    supervision_mode=args.supervision_mode,
                    band_selection=getattr(args, 'band_selection', 'all'),
                    normalize_bands=True
                )
                # Determine input channels from the loaded features
                in_channels = first_patch_features.shape[0]
                print(f"📊 Detected {in_channels} input channels from {args.band_selection} band selection")
                
                model, train_metrics = train_2d_unet(
                    first_patch_features, first_patch_target,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    base_channels=args.base_channels,
                    huber_delta=args.huber_delta,
                    shift_radius=args.shift_radius
                )
            model_path = os.path.join(args.output_dir, 'multi_patch_2d_unet_model.pth')

        
        if TORCH_AVAILABLE:
            torch.save(model.state_dict(), model_path)
        print(f"💾 Saved {args.model.upper()} model to: {model_path}")
        
        importance_data = {}
    
    # Save training metrics
    metrics_file = os.path.join(args.output_dir, 'multi_patch_training_metrics.json')
    save_training_metrics(train_metrics, importance_data, metrics_file)
    
    # Generate predictions for all patches if requested
    if args.generate_prediction:
        print("🔮 Generating predictions for all patches...")
        
        patch_predictions = {}
        
        for i, patch_info in enumerate(tqdm(patches, desc="Generating predictions")):
            try:
                prediction_array = generate_patch_prediction(
                    model, patch_info, args.model, is_temporal,
                    supervision_mode=getattr(args, 'supervision_mode', 'gedi'),
                    band_selection=getattr(args, 'band_selection', 'all')
                )
                
                # Save individual patch prediction
                patch_pred_filename = f"prediction_{args.model}_{patch_info.patch_id}.tif"
                patch_pred_path = os.path.join(args.output_dir, patch_pred_filename)
                
                # Save with same georeference as input patch
                with rasterio.open(patch_info.file_path) as src:
                    profile = src.profile.copy()
                    profile.update(count=1, dtype='float32')
                    
                    with rasterio.open(patch_pred_path, 'w', **profile) as dst:
                        dst.write(prediction_array.astype('float32'), 1)
                
                patch_predictions[patch_info.patch_id] = patch_pred_path
                
            except Exception as e:
                print(f"⚠️  Error generating prediction for {patch_info.patch_id}: {e}")
                continue
        
        print(f"✅ Generated predictions for {len(patch_predictions)}/{len(patches)} patches")
        
        # Create spatial mosaic if requested
        if (args.merge_predictions or args.create_spatial_mosaic) and patch_predictions:
            print(f"🔗 Creating spatial mosaic using '{args.merge_strategy}' strategy...")
            
            if USE_ENHANCED_MERGER:
                # Use enhanced spatial merger with improved NaN handling
                merger = EnhancedSpatialMerger(merge_strategy=args.merge_strategy)
                
                # Determine output filename
                if hasattr(args, 'mosaic_name') and args.mosaic_name:
                    if not args.mosaic_name.endswith('.tif'):
                        mosaic_filename = f"{args.mosaic_name}.tif"
                    else:
                        mosaic_filename = args.mosaic_name
                else:
                    mosaic_filename = f'spatial_mosaic_{args.model}.tif'
                
                merged_output_path = os.path.join(args.output_dir, mosaic_filename)
                
                merged_path = merger.merge_predictions_from_files(
                    patch_predictions, merged_output_path
                )
            else:
                # Fallback to original merger
                merger = PredictionMerger(patches, merge_strategy=args.merge_strategy)
                merged_output_path = os.path.join(args.output_dir, f'merged_prediction_{args.model}.tif')
                
                merged_path = merger.merge_predictions_from_files(
                    patch_predictions, merged_output_path
                )
            
            print(f"🗺️  Spatial mosaic saved to: {merged_path}")
    
    print("🎉 Multi-patch training completed successfully!")


def parse_args():
    
    parser = argparse.ArgumentParser(description='Unified Patch-Based Canopy Height Model Training and Prediction')
    
    # Input modes - single patch, multi-patch directory, or file list
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--patch-dir', type=str,
                           help='Directory containing multiple patch TIF files')

    # Model selection
    parser.add_argument('--model', type=str, required=True,
                       choices=['rf', 'mlp', '2d_unet',  'shift_aware_unet'],
                       help='Model type to train (shift_aware_unet includes geolocation compensation)')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for models and predictions')
    
    # Multi-patch specific options
    parser.add_argument('--patch-pattern', type=str, default='*.tif',
                       help='File pattern to match patches (e.g., "*_temporal_*.tif")')
    parser.add_argument('--merge-predictions', action='store_true',
                       help='Merge individual patch predictions into single GeoTIFF')
    parser.add_argument('--merge-strategy', type=str, default='average',
                       choices=['average', 'maximum', 'first'],
                       help='Strategy for merging overlapping predictions')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs for neural networks')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate for neural networks')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for neural networks')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data to use for validation')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay for neural networks')
    
    # Model parameters
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of estimators for Random Forest')
    parser.add_argument('--max-depth', type=int, default=10,
                       help='Maximum depth for Random Forest')
    parser.add_argument('--hidden-layers', type=str, default='128,64',
                       help='Hidden layer sizes for MLP (comma-separated)')
    parser.add_argument('--base-channels', type=int, default=32,
                       help='Base channels for U-Net models')
    parser.add_argument('--huber-delta', type=float, default=1.0,
                       help='Delta parameter for Huber loss')
    parser.add_argument('--shift-radius', type=int, default=1,
                       help='Spatial shift radius for GEDI alignment')
    
    # Enhanced Training Options (New Features)
    parser.add_argument('--augment', action='store_true',
                       help='Enable data augmentation (12x spatial transformations)')
    parser.add_argument('--augment-factor', type=int, default=12,
                       help='Number of augmentations per patch (default: 12)')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Fraction of patches for validation (default: 0.2)')
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                       help='Epochs to wait before early stopping (default: 15)')
    parser.add_argument('--checkpoint-freq', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--resume-from', type=str,
                       help='Resume training from checkpoint file')
    parser.add_argument('--use-enhanced-training', action='store_true',
                       help='Use enhanced training pipeline with full multi-patch support')
    parser.add_argument('--band-selection', type=str, default='all',
                       choices=['all', 'embedding', 'original', 'auxiliary'],
                       help='Band selection: all, embedding (A00-A63), original (30-band), auxiliary')
    
    # Prediction options
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'predict'],
                       help='Operation mode: train models or generate predictions only')
    parser.add_argument('--model-path', type=str,
                       help='Path to pre-trained model for prediction mode')
    parser.add_argument('--generate-prediction', action='store_true',
                       help='Generate prediction maps after training')
    # parser.add_argument('--prediction-output', type=str, default=None,
    #                    help='Output path for prediction TIF (auto-generated if not specified)')
    parser.add_argument('--save-model', action='store_true',
                       help='Save trained model to disk')
    parser.add_argument('--create-spatial-mosaic', action='store_true',
                       help='Create proper spatial mosaic (same as --merge-predictions but clearer name)')
    parser.add_argument('--mosaic-name', type=str, default=None,
                       help='Custom name for spatial mosaic output file')
    
    # GEDI filtering options
    parser.add_argument('--min-gedi-samples', type=int, default=10,
                       help='Minimum number of valid GEDI samples per patch for training (default: 10)')
    
    # Reference height supervision options
    parser.add_argument('--reference-height-path', type=str,
                       help='Path to reference height TIF file for dense supervision')
    parser.add_argument('--supervision-mode', type=str, default='gedi',
                       choices=['gedi', 'reference', 'reference_only'],
                       help='Supervision mode: gedi (sparse), reference (dense), reference_only (no GEDI)')
    parser.add_argument('--min-reference-samples', type=int, default=100,
                       help='Minimum number of valid reference samples per patch (default: 100)')
    # parser.add_argument('--use-augmentation', action='store_true',
    #                    help='Enable data augmentation (12x increase with flips + rotations)')
    # Pre-trained model support for fine-tuning
    parser.add_argument('--pretrained-model-path', type=str,
                       help='Path to pre-trained model for fine-tuning')
    parser.add_argument('--fine-tune-mode', action='store_true',
                       help='Enable fine-tuning mode (load pre-trained weights)')
    
    # Verbose output
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    return args

# def parse_args():
#     parser = argparse.ArgumentParser(description='Unified patch-based training for all model types')
    
#     # Input modes - either single patch or multi-patch directory
#     input_group = parser.add_mutually_exclusive_group(required=True)
#     input_group.add_argument('--patch-dir', type=str,
#                            help='Directory containing multiple patch TIF files')
    
#     # Multi-patch options
#     parser.add_argument('--patch-pattern', type=str, default='*.tif',
#                        help='File pattern for multi-patch mode (e.g., "*_temporal_*.tif")')
#     parser.add_argument('--merge-predictions', action='store_true',
#                        help='Merge individual patch predictions into continuous map')
#     parser.add_argument('--merge-strategy', type=str, default='first', 
#                        choices=['average', 'maximum', 'minimum', 'first', 'last'],
#                        help='Strategy for merging overlapping predictions')
#     parser.add_argument('--create-spatial-mosaic', action='store_true',
#                        help='Create proper spatial mosaic (same as --merge-predictions but clearer name)')
#     parser.add_argument('--mosaic-name', type=str, default=None,
#                        help='Custom name for spatial mosaic output file')
    
#     # Output settings
#     parser.add_argument('--output-dir', type=str, default='chm_outputs',
#                        help='Output directory for models and predictions')
    
#     # Model selection
#     parser.add_argument('--model', type=str, default='rf', choices=['rf', 'mlp', '2d_unet'],
#                        help='Model type: random forest (rf), MLP (mlp), 2D U-Net (2d_unet)')
    
#     # Traditional model parameters (RF/MLP)
#     parser.add_argument('--test-size', type=float, default=0.2,
#                        help='Proportion of GEDI pixels to use for validation (RF/MLP only)')
#     parser.add_argument('--batch-size', type=int, default=32,
#                        help='Batch size for training')
    
#     # Neural network parameters (all U-Nets)
#     parser.add_argument('--epochs', type=int, default=50,
#                        help='Number of training epochs (U-Net models)')
#     parser.add_argument('--learning-rate', type=float, default=1e-3,
#                        help='Learning rate (U-Net models)')
#     parser.add_argument('--weight-decay', type=float, default=1e-4,
#                        help='Weight decay (U-Net models)')
#     parser.add_argument('--base-channels', type=int, default=32,
#                        help='Base number of channels in U-Net models')
    
#     # Advanced training parameters
#     parser.add_argument('--huber-delta', type=float, default=1.0,
#                        help='Huber loss delta parameter (U-Net models)')
#     parser.add_argument('--shift-radius', type=int, default=1,
#                        help='Spatial shift radius for GEDI alignment (U-Net models)')
    
#     # Generation and evaluation
#     parser.add_argument('--generate-prediction', action='store_true',
#                        help='Generate prediction map after training')
#     parser.add_argument('--prediction-output', type=str, default=None,
#                        help='Output path for prediction TIF (auto-generated if not specified)')
    
#     # Training configuration - simple train/validation split
#     parser.add_argument('--validation-split', type=float, default=0.2,
#                        help='Fraction of patches to use for validation (default: 0.2)')
    
#     # Reference height supervision (Scenario 1: Reference-Only Training)
#     parser.add_argument('--reference-height-path', type=str, default=None,
#                        help='Path to reference height TIF for dense supervision (e.g., downloads/dchm_05LE4.tif)')
#     parser.add_argument('--supervision-mode', type=str, default='gedi', 
#                        choices=['gedi', 'reference', 'reference_only'],
#                        help='Supervision mode: gedi (sparse GEDI), reference_only (dense reference), reference (both)')
#     parser.add_argument('--min-reference-samples', type=int, default=1000,
#                        help='Minimum valid reference pixels per patch for training (reference supervision only)')
#     parser.add_argument('--min-gedi-samples', type=int, default=10,
#                        help='Minimum valid GEDI pixels per patch for training (GEDI supervision only)')
#     parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
#                        help='Mode: train (apply filtering) or predict (process all patches)')
#     parser.add_argument('--use-augmentation', action='store_true',
#                        help='Enable data augmentation (12x increase with flips + rotations)')
    
#     return parser.parse_args()

def main():
    """Unified main function for all model types using patch-based input."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    

    if args.verbose:
        print("🚀 Starting unified patch-based training system")
        print(f"Model: {args.model}")
        print(f"Output directory: {args.output_dir}")
    
    if args.patch_dir:
        # Multi-patch mode
        if args.verbose:
            print(f"📁 Multi-patch mode: {args.patch_dir}")
            print(f"🔍 Pattern: {args.patch_pattern}")
        
        try:
            # Check if enhanced training is requested for U-Net models
            if args.use_enhanced_training and args.model in ['2d_unet']:
                # Use enhanced U-Net training pipeline
                patch_files = glob.glob(os.path.join(args.patch_dir, args.patch_pattern))
                if not patch_files:
                    print(f"❌ No patches found matching pattern: {args.patch_pattern}")
                    exit(1)
                
                print(f"🚀 Using enhanced training pipeline for {args.model.upper()}")
                trainer = EnhancedUNetTrainer(model_type=args.model)
                
                training_results = trainer.train_multi_patch_unet(
                    patch_files=patch_files,
                    output_dir=args.output_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    validation_split=args.validation_split,
                    early_stopping_patience=args.early_stopping_patience,
                    augment=args.augment,
                    checkpoint_freq=args.checkpoint_freq,
                    supervision_mode=args.supervision_mode,
                    band_selection=getattr(args, 'band_selection', 'all')
                )
                
                print(f"🎉 Enhanced training completed!")
                print(f"📊 Results: {training_results}")
                
            else:
                # Use traditional training pipeline
                train_multi_patch(args)
                
        except Exception as e:
            print(f"❌ Error in multi-patch training: {e}")
            exit(1)
    
    
    print("🎉 Training completed successfully!")

if __name__ == "__main__":
    main()
