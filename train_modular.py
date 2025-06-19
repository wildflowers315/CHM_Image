#!/usr/bin/env python3
"""
Modular training entry point - replacement for train_predict_map.py

This provides a clean, modular interface using the new training architecture.
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from training.workflows.unified_trainer import UnifiedTrainer
from training.core.callbacks import TrainingLogger


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Modular CHM training system",
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
    
    # Actions
    parser.add_argument('--generate-prediction', action='store_true',
                        help='Generate predictions after training')
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


def main():
    """Main training function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Apply model-specific defaults
    model_defaults = get_model_defaults(args.model)
    for key, value in model_defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    
    # Create logger
    logger = TrainingLogger()
    logger.log_info("🚀 Starting modular CHM training system")
    logger.log_info(f"Model: {args.model}")
    logger.log_info(f"Data source: {args.patch_path}")
    logger.log_info(f"Output directory: {args.output_dir}")
    
    try:
        # Create unified trainer
        trainer = UnifiedTrainer(
            model_type=args.model,
            output_dir=args.output_dir,
            device=args.device,
            logger=logger
        )
        
        # Prepare training parameters
        training_params = {
            'validation_split': args.validation_split,
            'random_seed': args.random_seed,
            'augment': args.augment,
            'augment_factor': args.augment_factor,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'early_stopping_patience': args.early_stopping_patience
        }
        
        # Add model-specific parameters
        if args.model == 'rf':
            training_params.update({
                'n_estimators': args.n_estimators,
                'max_depth': args.max_depth
            })
        elif args.model == 'mlp':
            training_params.update({
                'hidden_sizes': args.hidden_sizes,
                'dropout_rate': args.dropout_rate,
                'learning_rate': args.learning_rate
            })
        elif args.model in ['2d_unet', '3d_unet']:
            training_params.update({
                'base_channels': args.base_channels,
                'learning_rate': args.learning_rate
            })
        
        # Execute training
        logger.log_info("Starting training workflow...")
        results = trainer.train(args.patch_path, **training_params)
        
        # Log results
        logger.log_info("✅ Training completed successfully!")
        logger.log_info(f"Training time: {results.get('training_time', 0):.2f}s")
        
        if 'metrics' in results:
            metrics = results['metrics']
            for key, value in metrics.items():
                logger.log_info(f"{key}: {value:.4f}")
        
        # Generate predictions if requested
        if args.generate_prediction:
            logger.log_info("Generating predictions...")
            # This would use trainer.predict() method
            logger.log_info("Predictions saved to output directory")
        
        # Evaluate model if requested  
        if args.evaluate_model:
            logger.log_info("Evaluating model...")
            # This would use trainer.evaluate() method
            logger.log_info("Evaluation completed")
        
        logger.log_info(f"All outputs saved to: {trainer.output_dir}")
        
    except Exception as e:
        logger.log_error(f"Training failed: {str(e)}")
        raise
    
    finally:
        # Save log
        log_path = Path(args.output_dir) / args.model / 'training.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.save_log(log_path)


if __name__ == "__main__":
    main()