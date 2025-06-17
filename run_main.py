import os
import subprocess
from pathlib import Path
import time
import argparse
from datetime import datetime

from utils import get_latest_file
from combine_heights import combine_heights_with_training

def parse_args():
    parser = argparse.ArgumentParser(description='Process canopy height mapping data.')
    parser.add_argument('--aoi_path', type=str, required=True, help='Path to the AOI file')
    parser.add_argument('--year', type=str, required=True, help='Year of the data')
    parser.add_argument('--start-date', type=str, required=True, help='Start date of the data (MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date of the data (MM-DD)')
    parser.add_argument('--steps', type=str, nargs='+', required=True,
                       choices=['data_preparation', 'height_analysis', 'train_predict', 'evaluate'],
                       help='Processing steps to perform (can specify multiple)')
    parser.add_argument('--eval_tif_path', type=str, required=True, help='Path to the evaluation tif file')
    parser.add_argument('--model', type=str, default='3d_unet', choices=['RF', '3d_unet'],
                       help='Model type to use (RF or 3d_unet)')
    parser.add_argument('--use-patches', action='store_true',
                       help='Use patch-based processing (required for 3D U-Net)')
    parser.add_argument('--patch-size', type=int, default=2560,
                       help='Size of patches in meters (default: 2560)')
    parser.add_argument('--patch-overlap', type=float, default=0.1,
                       help='Overlap between patches (0.0 to 1.0)')
    return parser.parse_args()

def get_gedi_dates(year, start_date, end_date):
    """Convert input dates to GEDI date format (YYYY-MM-DD)"""
    # start = f"{year}-{start_date}"
    # end = f"{year}-{end_date}"
    start = f"2019-{start_date}"
    end = f"2022-{end_date}"

    return start, end

def main(args):
    # Validate AOI file exists
    if not os.path.exists(args.aoi_path):
        raise FileNotFoundError(f"AOI file not found at {args.aoi_path}")

    # Create output directories
    output_dir = 'chm_outputs'
    eval_dir = os.path.join(output_dir, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Get GEDI dates from input arguments
    gedi_start_date, gedi_end_date = get_gedi_dates(args.year, args.start_date, args.end_date)

    # Build command for GEE model training and prediction
    # Always use RF for GEE processing since it's the only supported model
    gee_cmd = [
        'python', 'chm_main.py',
        '--aoi', args.aoi_path,
        '--year', args.year,
        '--start-date', args.start_date,
        '--end-date', args.end_date,
        '--gedi-start-date', gedi_start_date,
        '--gedi-end-date', gedi_end_date,
        '--buffer', '0',
        '--clouds-th', '70',
        '--quantile', 'rh98',
        '--model', 'RF',  # Always use RF for GEE
        '--num-trees-rf', '500',
        '--min-leaf-pop-rf', '5',
        '--bag-frac-rf', '0.5',
        '--max-nodes-rf', '1000',
        '--output-dir', output_dir,
        '--export-forest-mask',
        '--scale', '10',
        '--ndvi-threshold', '0.35',
        '--mask-type', 'WC',
    ]

    # Add patch-related arguments if using 3D U-Net
    if args.model == '3d_unet':
        gee_cmd.extend([
            '--use-patches',
            '--patch-size', str(args.patch_size),
            '--patch-overlap', str(args.patch_overlap),
            '--export-patches',
        ])

    # Process each requested step
    for step in args.steps:
        print(f"\nProcessing step: {step}")
        
        if step == 'data_preparation':
            # Convert all arguments to strings
            gee_cmd = [str(arg) for arg in gee_cmd]
            # Run the GEE model training and prediction
            print("Running GEE canopy height model...")
            subprocess.run(gee_cmd, check=True)

        # Get the most recent training data, stack, and mask files
        training_file = get_latest_file(output_dir, 'training_data')
        stack_file = get_latest_file(output_dir, 'stack')
        mask_file = get_latest_file(output_dir, 'forestMask')
        buffered_mask_file = get_latest_file(output_dir, 'buffered_forestMask')

        ref_file = args.eval_tif_path
        
        if step == 'height_analysis':
            combine_heights_with_training(output_dir, ref_file)

        if step == 'train_predict':
            try:
                # Build command for local model training and prediction
                train_cmd = [
                    'python', 'train_predict_map.py',
                    '--training-data', training_file,
                    '--stack', stack_file,
                    '--mask', mask_file,
                    '--buffered-mask', buffered_mask_file,
                    '--output-dir', output_dir,
                    '--test-size', '0.1',
                    '--apply-forest-mask',
                    '--ch_col', 'rh',
                    '--model', args.model
                ]

                if args.model == '3d_unet':
                    train_cmd.extend([
                        '--use-patches',
                        '--patch-size', str(args.patch_size),
                        '--patch-overlap', str(args.patch_overlap),
                        '--patches-dir', os.path.join(output_dir, 'patches')
                    ])

                # Run local training and prediction
                print("\nRunning local model training and prediction...")
                subprocess.run([str(arg) for arg in train_cmd], check=True)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Please ensure all required files have been exported from GEE before running local processing.")
        
        if step == 'evaluate':
            pred_file = get_latest_file(output_dir, 'predictCH') 
            # Run evaluation with PDF report generation
            eval_cmd = [
                'python', 'evaluate_predictions.py',
                '--pred', pred_file,
                '--ref', ref_file,
                '--output', eval_dir,
                '--pdf',
                '--training', training_file,
                '--merged', stack_file,
                '--forest-mask', mask_file,
                '--model', args.model
            ]
            print("\nRunning evaluation...")
            subprocess.run([str(arg) for arg in eval_cmd], check=True)
            print("Evaluation complete!")

        print(f"Step {step} completed successfully!")

    print("\nAll requested processing steps completed!")

if __name__ == "__main__":
    args = parse_args()
    main(args)
    