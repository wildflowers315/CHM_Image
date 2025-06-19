#!/usr/bin/env python3
"""
Simplified modular training script that works with existing code.
This bridges the gap while the full modular system is being completed.
"""

import argparse
import sys
import os
from pathlib import Path

def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Simplified modular CHM training system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--patch-path', required=True,
                        help='Path to patch file or directory containing patches')
    parser.add_argument('--model', required=True, 
                        choices=['rf', 'mlp', '2d_unet', '3d_unet'],
                        help='Model type to train')
    
    # Output options
    parser.add_argument('--output-dir', default='chm_outputs',
                        help='Base output directory')
    
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
    
    # Actions
    parser.add_argument('--generate-prediction', action='store_true',
                        help='Generate predictions after training')
    
    return parser

def construct_original_command(args):
    """Construct command for the original train_predict_map.py script."""
    cmd_parts = ['python', 'train_predict_map.py']
    
    # Required arguments
    cmd_parts.extend(['--patch-path', args.patch_path])
    cmd_parts.extend(['--model', args.model])
    cmd_parts.extend(['--output-dir', os.path.join(args.output_dir, args.model, 'predictions')])
    
    # Always use enhanced training (it's now fixed)
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
    
    # Set device if not auto
    if args.device != 'auto':
        if args.device == 'cuda':
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        elif args.device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    return cmd_parts

def main():
    """Main function that delegates to the original script."""
    parser = create_parser()
    args = parser.parse_args()
    
    print("🚀 Starting simplified modular CHM training system")
    print(f"Model: {args.model}")
    print(f"Data source: {args.patch_path}")
    print(f"Output directory: {args.output_dir}/{args.model}")
    print("📝 Delegating to fixed train_predict_map.py...")
    print()
    
    # Construct and execute command
    cmd = construct_original_command(args)
    cmd_str = ' '.join(cmd)
    
    print(f"Executing: {cmd_str}")
    print("=" * 80)
    
    # Execute the command
    exit_code = os.system(cmd_str)
    
    print("=" * 80)
    if exit_code == 0:
        print("✅ Training completed successfully!")
        print(f"📁 Outputs saved to: {args.output_dir}/{args.model}/")
    else:
        print("❌ Training failed!")
        sys.exit(exit_code)

if __name__ == "__main__":
    main()