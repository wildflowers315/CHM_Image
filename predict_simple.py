#!/usr/bin/env python3
"""
Simple prediction script that uses the working train_predict_map.py approach
"""

import argparse
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Simple prediction using train_predict_map.py")
    parser.add_argument('--patch-path', required=True, help='Path to patches')
    parser.add_argument('--model', required=True, choices=['2d_unet', '3d_unet'], help='Model type')
    parser.add_argument('--pretrained-model', required=True, help='Path to pretrained model')
    parser.add_argument('--output-dir', default='predictions', help='Output directory')
    
    args = parser.parse_args()
    
    print("🚀 Simple prediction mode")
    print(f"Model: {args.model}")
    print(f"Patches: {args.patch_path}")
    print(f"Pretrained model: {args.pretrained_model}")
    
    # Check if path is directory or file
    patch_path = Path(args.patch_path)
    if patch_path.is_dir():
        patch_arg = f'--patch-dir "{args.patch_path}"'
    else:
        patch_arg = f'--patch-path "{args.patch_path}"'
    
    # Construct command using the working train_predict_map.py
    cmd = f'''python train_predict_map.py \\
        {patch_arg} \\
        --model {args.model} \\
        --output-dir "{args.output_dir}" \\
        --resume-from "{args.pretrained_model}" \\
        --generate-prediction \\
        --use-enhanced-training \\
        --merge-predictions \\
        --merge-strategy average'''
    
    print(f"\nExecuting: {cmd}")
    print("=" * 80)
    
    # Execute the command
    exit_code = os.system(cmd)
    
    print("=" * 80)
    if exit_code == 0:
        print("✅ Prediction completed successfully!")
        print(f"📁 Check outputs in: {args.output_dir}")
    else:
        print("❌ Prediction failed!")
        
if __name__ == "__main__":
    main()