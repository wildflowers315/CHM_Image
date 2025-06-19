#!/usr/bin/env python3
"""
Modular training entry point - replacement for train_predict_map.py

This provides a clean, modular interface that delegates to the fixed train_predict_map.py
while the full modular system is being completed.
"""

import argparse
import sys
import os
from pathlib import Path


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
                        help='Path to pretrained model for prediction-only mode')
    
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
    
    # Use first patch as primary
    cmd_parts.extend(['--patch-path', f'"{patch_files[0]}"'])
    
    # Add multi-patch support if needed
    if len(patch_files) > 1:
        # Create temporary file list
        os.makedirs(args.output_dir, exist_ok=True)
        patch_list_file = f"{args.output_dir}/patch_list.txt"
        
        with open(patch_list_file, 'w') as f:
            for patch_file in patch_files:
                f.write(f"{patch_file}\n")
        
        print(f"📝 Created patch list file: {patch_list_file}")
        # Note: train_predict_map.py would need to support --patch-list argument
        # For now, we'll process multi-patch by using the directory
        patch_dir = str(Path(patch_files[0]).parent)
        cmd_parts[-1] = f'"{patch_dir}"'  # Replace single file with directory
    
    # Add mode-specific arguments
    if args.mode == 'predict':
        # For prediction-only mode
        pretrained_model = find_pretrained_model(args)
        if not pretrained_model:
            raise ValueError("No pretrained model found. Specify --pretrained-model or train first.")
        cmd_parts.extend(['--pretrained-model', f'"{pretrained_model}"'])
        cmd_parts.append('--predict-only')
    elif args.mode == 'train':
        # For training-only mode
        cmd_parts.append('--train-only')
    else:  # train_predict
        # Default mode - train and predict
        cmd_parts.append('--generate-prediction')
    
    # Standard arguments
    cmd_parts.extend(['--model', args.model])
    cmd_parts.extend(['--output-dir', f'"{args.output_dir}/{args.model}/predictions"'])
    cmd_parts.append('--use-enhanced-training')
    
    # Add aggregation if requested
    if args.aggregate_predictions:
        cmd_parts.append('--aggregate-predictions')
        cmd_parts.extend(['--overlap-method', args.overlap_method])
    
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
            cmd_parts.extend(['--hidden-sizes'] + [str(x) for x in args.hidden_sizes])
        if args.dropout_rate != 0.2:
            cmd_parts.extend(['--dropout-rate', str(args.dropout_rate)])
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
        
        # Validate mode and pretrained model
        if args.mode == 'predict':
            pretrained_model = find_pretrained_model(args)
            if not pretrained_model:
                print("❌ Prediction mode requires a pretrained model!")
                print("   Options:")
                print("   1. Specify --pretrained-model /path/to/model.pt")
                print("   2. Train first using --mode train")
                print("   3. Use --mode train_predict to train and predict")
                sys.exit(1)
            print(f"✅ Using pretrained model: {pretrained_model}")
        
        print("📝 Delegating to fixed train_predict_map.py (negative stride issue resolved)...")
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
            print("✅ Training/Prediction completed successfully!")
            print(f"📁 Outputs saved to: {args.output_dir}/{args.model}/")
            
            if args.mode in ['predict', 'train_predict']:
                print("🗺️  Prediction files should be available in the predictions directory")
                
            if args.aggregate_predictions and len(patch_files) > 1:
                print("🔗 Check for aggregated prediction map in the output directory")
                
            print("💡 Note: Full modular architecture is available in training/ package")
        else:
            print("❌ Training/Prediction failed!")
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