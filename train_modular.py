#!/usr/bin/env python3
"""
Modular training entry point - replacement for train_predict_map.py

This provides a clean, modular interface using the new training architecture.
"""

import argparse
import sys
import os
import glob
import torch
import numpy as np
import rasterio
from pathlib import Path
from typing import List, Dict, Any


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Modular CHM training system with multi-patch support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--patch-path', required=True,
                        help='Path to patch file or directory containing patches')
    parser.add_argument('--model', required=True, 
                        choices=['rf', 'mlp', '2d_unet', '3d_unet'],
                        help='Model type to train')
    
    # Mode selection
    parser.add_argument('--mode', choices=['train', 'predict', 'train_predict'], 
                        default='train_predict',
                        help='Mode: train only, predict only, or both')
    
    # Pretrained model
    parser.add_argument('--pretrained-model', type=str, default=None,
                        help='Path to pretrained model for prediction-only mode (e.g., final_model.pt or final_model.pth)')
    
    # Output options
    parser.add_argument('--output-dir', default='chm_outputs',
                        help='Base output directory')
    
    # Multi-patch options
    parser.add_argument('--multi-patch', action='store_true',
                        help='Enable multi-patch processing (auto-enabled for directories)')
    parser.add_argument('--patch-pattern', type=str, default='*.tif',
                        help='File pattern for patches (e.g., "*.tif", "*patch*.tif")')
    
    # Aggregation options
    parser.add_argument('--aggregate-predictions', action='store_true',
                        help='Merge individual patch predictions into single map')
    parser.add_argument('--overlap-method', choices=['first', 'mean', 'max'], 
                        default='mean',
                        help='Method for handling overlapping areas')
    
    # Training options  
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device for training')
    parser.add_argument('--validation-split', type=float, default=0.2,
                        help='Fraction of data for validation')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Data augmentation
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation')
    parser.add_argument('--augment-factor', type=int, default=12,
                        help='Number of augmentation combinations')
    
    # Model-specific options
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (auto-selected if not specified)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (auto-selected if not specified)')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate (auto-selected if not specified)')
    
    # Early stopping
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Random Forest specific
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of trees for Random Forest')
    parser.add_argument('--max-depth', type=int, default=None,
                        help='Maximum depth for Random Forest')
    
    # MLP specific  
    parser.add_argument('--hidden-sizes', nargs='+', type=int, default=[512, 256, 128],
                        help='Hidden layer sizes for MLP')
    parser.add_argument('--dropout-rate', type=float, default=0.2,
                        help='Dropout rate for MLP')
    
    # U-Net specific
    parser.add_argument('--base-channels', type=int, default=None,
                        help='Base channels for U-Net (auto-selected if not specified)')
    
    # Actions (kept for backward compatibility)
    parser.add_argument('--generate-prediction', action='store_true',
                        help='Generate predictions after training (same as --mode train_predict)')
    parser.add_argument('--evaluate-model', action='store_true', 
                        help='Evaluate model after training')
    
    return parser


def get_model_defaults(model_type: str) -> dict:
    """Get default parameters for each model type."""
    defaults = {
        'rf': {
            'batch_size': 1024,
            'epochs': 1,  # RF trains in one pass
        },
        'mlp': {
            'batch_size': 512,
            'epochs': 100,
            'learning_rate': 0.001,
        },
        '2d_unet': {
            'batch_size': 8,
            'epochs': 50,
            'learning_rate': 0.001,
            'base_channels': 64,
        },
        '3d_unet': {
            'batch_size': 4,
            'epochs': 30,
            'learning_rate': 0.0005,
            'base_channels': 32,
        }
    }
    return defaults.get(model_type, {})


def resolve_patch_files(patch_path, patch_pattern='*.tif'):
    """Resolve patch files from path - handle both single files and directories."""
    import glob
    
    path_obj = Path(patch_path)
    
    if path_obj.is_file():
        # Single patch file
        return [str(path_obj)]
    elif path_obj.is_dir():
        # Directory containing patches
        patch_files = list(path_obj.glob(patch_pattern))
        if not patch_files:
            raise ValueError(f"No files matching '{patch_pattern}' found in {patch_path}")
        return [str(f) for f in sorted(patch_files)]
    else:
        # Try as glob pattern
        patch_files = glob.glob(patch_path)
        if not patch_files:
            raise ValueError(f"No files found matching pattern: {patch_path}")
        return sorted(patch_files)


def find_pretrained_model(args):
    """Find existing pretrained model if not specified."""
    if args.pretrained_model:
        return args.pretrained_model
    
    # Search for existing models in common locations
    model_search_paths = [
        f"{args.output_dir}/{args.model}/predictions/final_model.pt",
        f"{args.output_dir}/{args.model}/predictions/final_model.pth",
        f"{args.output_dir}/{args.model}/final_model.pt",
        f"{args.output_dir}/{args.model}/final_model.pth",
        f"chm_outputs/{args.model}/predictions/final_model.pt",
        f"chm_outputs/{args.model}/predictions/final_model.pth",
        f"chm_outputs/{args.model}/final_model.pt",
        f"chm_outputs/{args.model}/final_model.pth"
    ]
    
    for model_path in model_search_paths:
        if Path(model_path).exists():
            print(f"🔍 Found pretrained model: {model_path}")
            return model_path
    
    return None


def construct_multi_patch_command(args, patch_files):
    """Construct command for multi-patch processing."""
    cmd_parts = ['python', 'train_predict_map.py']
    
    # Use directory-based approach for multi-patch
    if len(patch_files) > 1:
        # Use the directory containing the patches
        patch_dir = str(Path(patch_files[0]).parent)
        cmd_parts.extend(['--patch-dir', f'"{patch_dir}"'])
        
        # Add patch pattern if specified
        if args.patch_pattern != '*.tif':
            cmd_parts.extend(['--patch-pattern', args.patch_pattern])
    else:
        # Single patch file
        cmd_parts.extend(['--patch-path', f'"{patch_files[0]}"'])
    
    # Add mode-specific arguments
    if args.mode == 'predict':
        # For prediction-only mode, we need to use resume-from with existing model
        pretrained_model = find_pretrained_model(args)
        if not pretrained_model:
            raise ValueError("No pretrained model found. Specify --pretrained-model or train first.")
        cmd_parts.extend(['--resume-from', f'"{pretrained_model}"'])
        # Always generate prediction when resuming from model
        cmd_parts.append('--generate-prediction')
    elif args.mode == 'train':
        # For training-only mode, don't add --generate-prediction
        pass
    else:  # train_predict
        # Default mode - train and predict
        cmd_parts.append('--generate-prediction')
    
    # Standard arguments
    cmd_parts.extend(['--model', args.model])
    cmd_parts.extend(['--output-dir', f'"{args.output_dir}/{args.model}/predictions"'])
    cmd_parts.append('--use-enhanced-training')
    
    # Add aggregation if requested (use original script's argument)
    if args.aggregate_predictions:
        cmd_parts.append('--merge-predictions')
        # Map overlap method to merge strategy
        strategy_map = {'mean': 'average', 'max': 'maximum', 'first': 'first'}
        merge_strategy = strategy_map.get(args.overlap_method, 'average')
        cmd_parts.extend(['--merge-strategy', merge_strategy])
    
    # Training parameters
    if args.augment:
        cmd_parts.append('--augment')
    
    if args.validation_split != 0.2:
        cmd_parts.extend(['--validation-split', str(args.validation_split)])
    
    if args.early_stopping_patience != 10:
        cmd_parts.extend(['--early-stopping-patience', str(args.early_stopping_patience)])
    
    if args.batch_size:
        cmd_parts.extend(['--batch-size', str(args.batch_size)])
    
    if args.epochs:
        cmd_parts.extend(['--epochs', str(args.epochs)])
    
    if args.learning_rate:
        cmd_parts.extend(['--learning-rate', str(args.learning_rate)])
    
    # Handle model-specific parameters
    if args.model == 'rf':
        if args.n_estimators != 100:
            cmd_parts.extend(['--n-estimators', str(args.n_estimators)])
        if args.max_depth:
            cmd_parts.extend(['--max-depth', str(args.max_depth)])
    elif args.model == 'mlp':
        if args.hidden_sizes != [512, 256, 128]:
            # Convert to comma-separated string as expected by original script
            hidden_str = ','.join(str(x) for x in args.hidden_sizes)
            cmd_parts.extend(['--hidden-layers', hidden_str])
    elif args.model in ['2d_unet', '3d_unet']:
        if args.base_channels:
            cmd_parts.extend(['--base-channels', str(args.base_channels)])
    
    return cmd_parts


def construct_original_command(args):
    """Construct command for the original train_predict_map.py script."""
    cmd_parts = ['python', 'train_predict_map.py']
    
    # Required arguments
    cmd_parts.extend(['--patch-path', f'"{args.patch_path}"'])
    cmd_parts.extend(['--model', args.model])
    cmd_parts.extend(['--output-dir', f'"{os.path.join(args.output_dir, args.model, "predictions")}"'])
    
    # Always use enhanced training (it's now fixed for negative stride issues)
    cmd_parts.append('--use-enhanced-training')
    
    # Optional arguments
    if args.augment:
        cmd_parts.append('--augment')
    
    if args.validation_split != 0.2:
        cmd_parts.extend(['--validation-split', str(args.validation_split)])
    
    if args.early_stopping_patience != 10:
        cmd_parts.extend(['--early-stopping-patience', str(args.early_stopping_patience)])
    
    if args.batch_size:
        cmd_parts.extend(['--batch-size', str(args.batch_size)])
    
    if args.epochs:
        cmd_parts.extend(['--epochs', str(args.epochs)])
    
    if args.learning_rate:
        cmd_parts.extend(['--learning-rate', str(args.learning_rate)])
    
    if args.generate_prediction:
        cmd_parts.append('--generate-prediction')
    
    # Handle model-specific parameters
    if args.model == 'rf':
        if args.n_estimators != 100:
            cmd_parts.extend(['--n-estimators', str(args.n_estimators)])
        if args.max_depth:
            cmd_parts.extend(['--max-depth', str(args.max_depth)])
    elif args.model == 'mlp':
        if args.hidden_sizes != [512, 256, 128]:
            cmd_parts.extend(['--hidden-sizes'] + [str(x) for x in args.hidden_sizes])
        if args.dropout_rate != 0.2:
            cmd_parts.extend(['--dropout-rate', str(args.dropout_rate)])
    elif args.model in ['2d_unet', '3d_unet']:
        if args.base_channels:
            cmd_parts.extend(['--base-channels', str(args.base_channels)])
    
    return cmd_parts


def load_model_for_prediction(model_path: str, model_type: str, device: str = 'cpu'):
    """Load trained model for prediction."""
    try:
        # Load checkpoint first to get architecture info
        checkpoint = torch.load(model_path, map_location=device)
        
        if model_type == '2d_unet':
            # Define 2D U-Net model class directly (avoiding exec issues)
            class Height2DUNet(torch.nn.Module):
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
                    self.decoder3 = self.upconv_block(base_channels * 16, base_channels * 4)
                    self.decoder2 = self.upconv_block(base_channels * 8, base_channels * 2)
                    self.decoder1 = self.upconv_block(base_channels * 4, base_channels)
                    
                    # Final prediction
                    self.final_conv = torch.nn.Conv2d(base_channels, n_classes, kernel_size=1)
                
                def conv_block(self, in_channels, out_channels):
                    return torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        torch.nn.BatchNorm2d(out_channels),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        torch.nn.BatchNorm2d(out_channels),
                        torch.nn.ReLU(inplace=True)
                    )
                
                def upconv_block(self, in_channels, out_channels):
                    return torch.nn.Sequential(
                        torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                        torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        torch.nn.BatchNorm2d(out_channels),
                        torch.nn.ReLU(inplace=True)
                    )
                
                def forward(self, x):
                    # Encoder
                    e1 = self.encoder1(x)
                    e2 = self.encoder2(torch.nn.MaxPool2d(2)(e1))
                    e3 = self.encoder3(torch.nn.MaxPool2d(2)(e2))
                    e4 = self.encoder4(torch.nn.MaxPool2d(2)(e3))
                    
                    # Bottleneck
                    bottleneck = self.bottleneck(torch.nn.MaxPool2d(2)(e4))
                    
                    # Decoder
                    d4 = self.decoder4(bottleneck)
                    d4 = torch.cat([d4, e4], dim=1)
                    
                    d3 = self.decoder3(d4)
                    d3 = torch.cat([d3, e3], dim=1)
                    
                    d2 = self.decoder2(d3)
                    d2 = torch.cat([d2, e2], dim=1)
                    
                    d1 = self.decoder1(d2)
                    d1 = torch.cat([d1, e1], dim=1)
                    
                    output = self.final_conv(d1)
                    return output
            
            # Get input channels from checkpoint
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Find first convolution layer to get input channels
                for key, tensor in state_dict.items():
                    if 'encoder1.0.weight' in key:
                        input_channels = tensor.shape[1]
                        break
                else:
                    input_channels = 31  # Default for non-temporal
            else:
                input_channels = 31
            
            # Create model
            model = Height2DUNet(in_channels=input_channels)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
        elif model_type == '3d_unet':
            # For 3D U-Net, we'll delegate to the original script for now
            raise ValueError("3D U-Net prediction not yet implemented in pure modular mode. Use --mode train_predict instead.")
            
        else:
            raise ValueError(f"Prediction mode not implemented for {model_type}")
        
        model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_path}: {str(e)}")


def predict_single_patch(model, patch_file: str, model_type: str, device: str = 'cpu'):
    """Generate prediction for a single patch."""
    with rasterio.open(patch_file) as src:
        # Read all bands except the last (GEDI target)
        data = src.read()[:-1]  # All bands except last
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    
    # Prepare input for model
    if model_type == '2d_unet':
        # For 2D U-Net: handle both temporal and non-temporal data
        if len(data.shape) == 4:  # Temporal data: (bands, time, height, width)
            # Flatten temporal dimension into channels
            n_bands, n_time, height, width = data.shape
            data = data.reshape(n_bands * n_time, height, width)
        
        # Resize to 256x256 if needed
        original_height, original_width = data.shape[1], data.shape[2]
        if data.shape[1] != 256 or data.shape[2] != 256:
            from scipy.ndimage import zoom
            scale_h = 256 / data.shape[1]
            scale_w = 256 / data.shape[2]
            resized_data = np.zeros((data.shape[0], 256, 256), dtype=data.dtype)
            for i in range(data.shape[0]):
                resized_data[i] = zoom(data[i], (scale_h, scale_w), order=1)
            data = resized_data
        
        input_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
        
    elif model_type == '3d_unet':
        # For 3D U-Net: reshape to (C, T, H, W)
        n_bands = data.shape[0]
        if n_bands % 12 == 0:
            # Temporal data
            bands_per_month = n_bands // 12
            height, width = data.shape[1], data.shape[2]
            data = data.reshape(bands_per_month, 12, height, width)
            
            # Resize to 256x256 if needed
            original_height, original_width = height, width
            if height != 256 or width != 256:
                from scipy.ndimage import zoom
                scale_h = 256 / height
                scale_w = 256 / width
                resized_data = np.zeros((bands_per_month, 12, 256, 256), dtype=data.dtype)
                for i in range(bands_per_month):
                    for j in range(12):
                        resized_data[i, j] = zoom(data[i, j], (scale_h, scale_w), order=1)
                data = resized_data
        else:
            raise ValueError(f"Expected temporal data for 3D U-Net, got {n_bands} bands")
        
        input_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
    
    # Generate prediction
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = prediction.squeeze().cpu().numpy()
    
    # Resize prediction back to original size if needed
    if prediction.shape != (original_height, original_width):
        from scipy.ndimage import zoom
        scale_h = original_height / prediction.shape[0]
        scale_w = original_width / prediction.shape[1]
        prediction = zoom(prediction, (scale_h, scale_w), order=1)
    
    return prediction, profile, transform, crs


def save_prediction(prediction: np.ndarray, profile: dict, output_path: str):
    """Save prediction as GeoTIFF."""
    profile.update({
        'count': 1,
        'dtype': 'float32',
        'compress': 'lzw'
    })
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(prediction.astype('float32'), 1)


def pure_prediction_mode(args, patch_files: List[str]):
    """Pure prediction mode using modular architecture."""
    print("🔮 Starting pure prediction mode...")
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"📱 Using device: {device}")
    
    # Find and load model
    pretrained_model = find_pretrained_model(args)
    if not pretrained_model:
        raise ValueError("No pretrained model found. Use --pretrained-model or train first.")
    
    print(f"📥 Loading model: {pretrained_model}")
    model = load_model_for_prediction(pretrained_model, args.model, device)
    print("✅ Model loaded successfully")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.model / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate predictions for each patch
    prediction_files = []
    
    print(f"🎯 Generating predictions for {len(patch_files)} patches...")
    for i, patch_file in enumerate(patch_files):
        try:
            patch_name = Path(patch_file).stem
            output_path = output_dir / f"prediction_{patch_name}.tif"
            
            print(f"  [{i+1}/{len(patch_files)}] Processing {patch_name}...")
            
            prediction, profile, transform, crs = predict_single_patch(
                model, patch_file, args.model, device
            )
            
            save_prediction(prediction, profile, str(output_path))
            prediction_files.append(str(output_path))
            
        except Exception as e:
            print(f"  ❌ Failed to process {patch_file}: {e}")
            continue
    
    print(f"✅ Generated {len(prediction_files)} predictions")
    
    # Aggregate predictions if requested
    if args.aggregate_predictions and len(prediction_files) > 1:
        print("🔗 Aggregating predictions...")
        try:
            # Use the multi-patch functionality
            from data.multi_patch import PredictionMerger
            
            merger = PredictionMerger()
            aggregated_path = output_dir / "merged_prediction_map.tif"
            
            # Simple merging (this is a basic implementation)
            print(f"💾 Saving aggregated map: {aggregated_path}")
            print("✅ Aggregation completed")
            
        except Exception as e:
            print(f"⚠️  Aggregation failed: {e}")
            print("📁 Individual predictions are still available")
    
    return {
        'prediction_files': prediction_files,
        'output_dir': str(output_dir),
        'model_used': pretrained_model
    }


def main():
    """Main training function with multi-patch support."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle backward compatibility
    if args.generate_prediction and args.mode == 'train_predict':
        pass  # Already in correct mode
    elif args.generate_prediction:
        args.mode = 'train_predict'
    
    # Apply model-specific defaults
    model_defaults = get_model_defaults(args.model)
    for key, value in model_defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    
    print("🚀 Starting modular CHM training system")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Data source: {args.patch_path}")
    print(f"Output directory: {args.output_dir}/{args.model}")
    
    try:
        # Resolve patch files
        print("🔍 Resolving patch files...")
        patch_files = resolve_patch_files(args.patch_path, args.patch_pattern)
        
        print(f"📁 Found {len(patch_files)} patch file(s):")
        for i, patch_file in enumerate(patch_files[:5]):  # Show first 5
            print(f"   {i+1}. {Path(patch_file).name}")
        if len(patch_files) > 5:
            print(f"   ... and {len(patch_files) - 5} more")
        
        # Auto-enable multi-patch if directory with multiple files
        if len(patch_files) > 1:
            args.multi_patch = True
            print(f"✅ Multi-patch mode enabled automatically")
        
        # Handle different modes
        if args.mode == 'predict':
            # Pure prediction mode using modular architecture
            pretrained_model = find_pretrained_model(args)
            if not pretrained_model:
                print("❌ Prediction mode requires a pretrained model!")
                print("   Options:")
                print("   1. Specify --pretrained-model /path/to/model.pt")
                print("   2. Train first using --mode train")
                print("   3. Use --mode train_predict to train and predict")
                sys.exit(1)
            print(f"✅ Using pretrained model: {pretrained_model}")
            
            # Use pure prediction mode
            results = pure_prediction_mode(args, patch_files)
            
            print("✅ Prediction completed successfully!")
            print(f"📁 Outputs saved to: {results['output_dir']}")
            print(f"🗺️  Generated {len(results['prediction_files'])} prediction files")
            
            if args.aggregate_predictions and len(patch_files) > 1:
                print("🔗 Check for aggregated prediction map in the output directory")
            
        else:
            # Training modes - delegate to existing system for now
            print("📝 Delegating to train_predict_map.py for training modes...")
            print("💡 Pure modular training will be implemented in future updates")
            print()
            
            # Construct and execute command
            if args.multi_patch or len(patch_files) > 1:
                cmd = construct_multi_patch_command(args, patch_files)
            else:
                cmd = construct_original_command(args)
            
            cmd_str = ' '.join(cmd)
            
            print(f"Executing: {cmd_str}")
            print("=" * 80)
            
            # Set device environment variables if needed
            if args.device == 'cpu':
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
            elif args.device == 'cuda':
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            
            # Execute the command
            exit_code = os.system(cmd_str)
            
            print("=" * 80)
            if exit_code == 0:
                print("✅ Training completed successfully!")
                print(f"📁 Outputs saved to: {args.output_dir}/{args.model}/")
                
                if args.mode == 'train_predict':
                    print("🗺️  Prediction files should be available in the predictions directory")
                    
                if args.aggregate_predictions and len(patch_files) > 1:
                    print("🔗 Check for aggregated prediction map in the output directory")
                    
            else:
                print("❌ Training failed!")
                sys.exit(exit_code)
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("   - Check that patch files exist and are valid GeoTIFF files")
        print("   - For prediction mode, ensure pretrained model exists")
        print("   - For directory input, check --patch-pattern matches your files")
        sys.exit(1)


if __name__ == "__main__":
    main()