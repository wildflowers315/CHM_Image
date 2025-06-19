#!/usr/bin/env python3
"""
Unified evaluation system for train_predict_map.py outputs
Generates comprehensive PDF reports with model evaluation and analysis
"""

import os
import json
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
from datetime import datetime
import glob
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import warnings
from dataclasses import dataclass
import joblib

warnings.filterwarnings('ignore')

# Set style for professional plots - use fallback if seaborn style not available
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('default')
    
try:
    sns.set_palette("husl")
except:
    pass  # Continue without seaborn styling if not available

@dataclass
class EvaluationResults:
    """Container for evaluation results and metadata."""
    output_dir: str
    model_type: str
    training_metrics: Dict[str, float]
    prediction_files: List[str]
    patch_summary: Optional[pd.DataFrame] = None
    feature_importance: Optional[Dict[str, float]] = None
    model_info: Optional[Dict[str, Any]] = None
    temporal_mode: bool = False
    
class UnifiedEvaluationSystem:
    """Comprehensive evaluation system for train_predict_map.py outputs."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.results = None
        
    def load_training_results(self) -> EvaluationResults:
        """Load all training results and metadata from output directory."""
        
        print(f"🔍 Loading results from: {self.output_dir}")
        
        # Load training metrics
        metrics_file = self.output_dir / "multi_patch_training_metrics.json"
        if not metrics_file.exists():
            # Try single patch metrics
            metrics_file = self.output_dir / "training_metrics.json"
            
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
                training_metrics = metrics_data.get('training_metrics', {})
                feature_importance = metrics_data.get('feature_importance', {})
        else:
            print(f"⚠️  No metrics file found")
            training_metrics = {}
            feature_importance = {}
            
        # Load patch summary if available
        patch_summary = None
        summary_file = self.output_dir / "patch_summary.csv"
        if summary_file.exists():
            patch_summary = pd.read_csv(summary_file)
            print(f"✅ Loaded patch summary: {len(patch_summary)} patches")
            
        # Find prediction files
        prediction_files = []
        for pattern in ['*.tif', '*.tiff']:
            pred_files = list(self.output_dir.glob(pattern))
            prediction_files.extend([str(f) for f in pred_files])
            
        if not prediction_files:
            print(f"⚠️  No prediction files found")
            
        # Determine model type from files (check both prediction files and model files)
        model_type = "Unknown"
        all_files = prediction_files + [str(f) for f in self.output_dir.glob("*")]
        
        if any("rf_" in str(f).lower() for f in all_files):
            model_type = "Random Forest"
        elif any("mlp_" in str(f).lower() for f in all_files):
            model_type = "MLP"
        elif any("2dunet_" in str(f).lower() for f in all_files):
            model_type = "2D U-Net"
        elif any("3dunet_" in str(f).lower() for f in all_files):
            model_type = "3D U-Net"
            
        # Determine temporal mode
        temporal_mode = False
        if patch_summary is not None:
            temporal_mode = patch_summary['temporal_mode'].any()
        elif "temporal" in str(self.output_dir).lower():
            temporal_mode = True
            
        # Load model info if available
        model_info = {}
        model_files = list(self.output_dir.glob("*.pkl")) + list(self.output_dir.glob("*.pth"))
        if model_files:
            model_info['model_files'] = [str(f) for f in model_files]
            model_info['model_count'] = len(model_files)
            
        self.results = EvaluationResults(
            output_dir=str(self.output_dir),
            model_type=model_type,
            training_metrics=training_metrics,
            prediction_files=prediction_files,
            patch_summary=patch_summary,
            feature_importance=feature_importance,
            model_info=model_info,
            temporal_mode=temporal_mode
        )
        
        print(f"✅ Loaded {model_type} results ({'temporal' if temporal_mode else 'non-temporal'})")
        print(f"   Training metrics: {len(training_metrics)} metrics")
        print(f"   Predictions: {len(prediction_files)} files")
        
        return self.results
        
    def analyze_predictions(self) -> Dict[str, Any]:
        """Analyze prediction files and extract statistics."""
        
        if not self.results:
            raise ValueError("Must load training results first")
            
        analysis = {
            'spatial_stats': {},
            'prediction_quality': {},
            'coverage_analysis': {}
        }
        
        print(f"🔬 Analyzing {len(self.results.prediction_files)} prediction files...")
        
        for pred_file in self.results.prediction_files:
            try:
                with rasterio.open(pred_file) as src:
                    prediction = src.read(1)
                    transform = src.transform
                    crs = src.crs
                    
                # Basic statistics
                valid_mask = ~np.isnan(prediction) & (prediction > 0)
                valid_data = prediction[valid_mask]
                
                if len(valid_data) > 0:
                    stats = {
                        'mean_height': float(valid_data.mean()),
                        'std_height': float(valid_data.std()),
                        'min_height': float(valid_data.min()),
                        'max_height': float(valid_data.max()),
                        'coverage_percent': float(valid_mask.sum() / valid_mask.size * 100),
                        'total_pixels': int(valid_mask.size),
                        'valid_pixels': int(valid_mask.sum()),
                        'shape': prediction.shape,
                        'crs': str(crs)
                    }
                    
                    # Height distribution analysis
                    stats['height_percentiles'] = {
                        'p25': float(np.percentile(valid_data, 25)),
                        'p50': float(np.percentile(valid_data, 50)),
                        'p75': float(np.percentile(valid_data, 75)),
                        'p90': float(np.percentile(valid_data, 90)),
                        'p95': float(np.percentile(valid_data, 95))
                    }
                    
                    analysis['spatial_stats'][Path(pred_file).name] = stats
                    
            except Exception as e:
                print(f"⚠️  Error analyzing {pred_file}: {e}")
                
        return analysis
        
    def generate_pdf_report(self, output_path: str = None) -> str:
        """Generate comprehensive PDF evaluation report."""
        
        if not self.results:
            raise ValueError("Must load training results first")
            
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"evaluation_report_{timestamp}.pdf"
        
        # Analyze predictions
        pred_analysis = self.analyze_predictions()
        
        print(f"📊 Generating PDF report: {output_path}")
        
        with PdfPages(output_path) as pdf:
            # Page 1: Overview and Summary
            self._create_overview_page(pdf, pred_analysis)
            
            # Page 2: Training Metrics
            self._create_training_metrics_page(pdf)
            
            # Page 3: Feature Importance (if available)
            if self.results.feature_importance:
                self._create_feature_importance_page(pdf)
                
            # Page 4: Spatial Analysis
            self._create_spatial_analysis_page(pdf, pred_analysis)
            
            # Page 5: Prediction Visualizations
            self._create_prediction_visualizations_page(pdf)
            
            # Page 6: Multi-patch Analysis (if applicable)
            if self.results.patch_summary is not None:
                self._create_multipatch_analysis_page(pdf)
                
        print(f"✅ PDF report saved: {output_path}")
        return str(output_path)
        
    def _create_overview_page(self, pdf: PdfPages, pred_analysis: Dict):
        """Create overview page with summary information."""
        
        fig = plt.figure(figsize=(8.5, 11))
        gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Title
        fig.suptitle(f'Canopy Height Model Evaluation Report\n{self.results.model_type} - {"Temporal" if self.results.temporal_mode else "Non-temporal"}', 
                    fontsize=16, fontweight='bold')
        
        # Summary statistics table
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('tight')
        ax1.axis('off')
        
        summary_data = [
            ['Model Type', self.results.model_type],
            ['Processing Mode', 'Temporal' if self.results.temporal_mode else 'Non-temporal'],
            ['Output Directory', str(self.results.output_dir)],
            ['Prediction Files', len(self.results.prediction_files)],
            ['Report Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        
        if self.results.patch_summary is not None:
            summary_data.append(['Number of Patches', len(self.results.patch_summary)])
            
        table = ax1.table(cellText=summary_data, colLabels=['Parameter', 'Value'],
                         cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax1.set_title('Experiment Summary', fontsize=14, fontweight='bold', pad=20)
        
        # Training metrics summary
        if self.results.training_metrics:
            ax2 = fig.add_subplot(gs[1, 0])
            metrics = self.results.training_metrics
            
            key_metrics = ['RMSE', 'MAE', 'R2', 'Within 5m (%)']
            values = [metrics.get(k, 0) for k in key_metrics]
            
            bars = ax2.bar(range(len(key_metrics)), values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            ax2.set_xticks(range(len(key_metrics)))
            ax2.set_xticklabels(key_metrics, rotation=45)
            ax2.set_title('Key Training Metrics', fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9)
                        
        # Prediction coverage
        if pred_analysis['spatial_stats']:
            ax3 = fig.add_subplot(gs[1, 1])
            
            files = list(pred_analysis['spatial_stats'].keys())
            coverages = [pred_analysis['spatial_stats'][f]['coverage_percent'] for f in files]
            
            ax3.bar(range(len(files)), coverages, color='#FFA07A')
            ax3.set_xticks(range(len(files)))
            ax3.set_xticklabels([f[:15] + '...' if len(f) > 15 else f for f in files], 
                               rotation=45, ha='right')
            ax3.set_ylabel('Coverage (%)')
            ax3.set_title('Prediction Coverage', fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)
            
        # Height distribution overview
        if pred_analysis['spatial_stats']:
            ax4 = fig.add_subplot(gs[2:, :])
            
            all_stats = list(pred_analysis['spatial_stats'].values())
            if all_stats:
                heights_data = []
                labels = []
                
                for i, (fname, stats) in enumerate(pred_analysis['spatial_stats'].items()):
                    if 'height_percentiles' in stats:
                        heights_data.append([
                            stats['height_percentiles']['p25'],
                            stats['height_percentiles']['p50'],
                            stats['height_percentiles']['p75'],
                            stats['height_percentiles']['p90']
                        ])
                        labels.append(fname[:20] + '...' if len(fname) > 20 else fname)
                        
                if heights_data:
                    bp = ax4.boxplot(heights_data, labels=labels, patch_artist=True)
                    
                    # Color the boxes
                    colors = ['#FFB6C1', '#87CEEB', '#98FB98', '#F0E68C']
                    for patch, color in zip(bp['boxes'], colors * len(heights_data)):
                        patch.set_facecolor(color)
                        
                    ax4.set_ylabel('Height (m)')
                    ax4.set_title('Height Distribution by Prediction File', fontweight='bold')
                    ax4.grid(axis='y', alpha=0.3)
                    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
                    
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    def _create_training_metrics_page(self, pdf: PdfPages):
        """Create training metrics analysis page."""
        
        fig = plt.figure(figsize=(8.5, 11))
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        fig.suptitle('Training Performance Analysis', fontsize=16, fontweight='bold')
        
        if not self.results.training_metrics:
            # No metrics available
            ax = fig.add_subplot(gs[1, :])
            ax.text(0.5, 0.5, 'No training metrics available', 
                   ha='center', va='center', fontsize=16)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            return
            
        metrics = self.results.training_metrics
        
        # Error metrics
        ax1 = fig.add_subplot(gs[0, 0])
        error_metrics = ['MSE', 'RMSE', 'MAE', 'Max Absolute Error']
        error_values = [metrics.get(m, 0) for m in error_metrics]
        
        bars1 = ax1.bar(range(len(error_metrics)), error_values, 
                       color=['#FF6B6B', '#FF8E53', '#FF6B9D', '#C44569'])
        ax1.set_xticks(range(len(error_metrics)))
        ax1.set_xticklabels(error_metrics, rotation=45, ha='right')
        ax1.set_ylabel('Error Value')
        ax1.set_title('Error Metrics', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars1, error_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
                    
        # Accuracy metrics
        ax2 = fig.add_subplot(gs[0, 1])
        accuracy_metrics = ['Within 1m (%)', 'Within 2m (%)', 'Within 5m (%)']
        accuracy_values = [metrics.get(m, 0) for m in accuracy_metrics]
        
        bars2 = ax2.bar(range(len(accuracy_metrics)), accuracy_values,
                       color=['#4ECDC4', '#44A08D', '#096A09'])
        ax2.set_xticks(range(len(accuracy_metrics)))
        ax2.set_xticklabels(['1m', '2m', '5m'])
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Distance Accuracy', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Add value labels
        for bar, val in zip(bars2, accuracy_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
                    
        # R² and correlation
        ax3 = fig.add_subplot(gs[1, 0])
        r2_value = metrics.get('R2', 0)
        
        # Create a gauge-like plot for R²
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        ax3.plot(theta, r, 'k-', linewidth=2)
        ax3.fill_between(theta, 0, r, alpha=0.3, color='lightgray')
        
        # Mark R² value
        r2_theta = np.pi * (1 - r2_value)  # Invert because R² close to 1 should be on right
        ax3.plot([r2_theta, r2_theta], [0, 1], 'r-', linewidth=3)
        ax3.plot(r2_theta, 1, 'ro', markersize=8)
        
        ax3.set_xlim(0, np.pi)
        ax3.set_ylim(0, 1.2)
        ax3.set_aspect('equal')
        ax3.set_title(f'R² Score: {r2_value:.3f}', fontweight='bold')
        ax3.text(0, -0.1, '0.0', ha='center')
        ax3.text(np.pi, -0.1, '1.0', ha='center')
        ax3.axis('off')
        
        # Error distribution
        ax4 = fig.add_subplot(gs[1, 1])
        if 'Mean Error' in metrics and 'Std Error' in metrics:
            mean_err = metrics['Mean Error']
            std_err = metrics['Std Error']
            
            # Create normal distribution curve
            x = np.linspace(mean_err - 3*std_err, mean_err + 3*std_err, 100)
            y = (1/(std_err * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean_err)/std_err)**2)
            
            ax4.plot(x, y, 'b-', linewidth=2, label=f'Error Distribution')
            ax4.axvline(mean_err, color='r', linestyle='--', 
                       label=f'Mean: {mean_err:.2f}m')
            ax4.axvline(0, color='g', linestyle='--', alpha=0.7, label='Perfect')
            ax4.fill_between(x, 0, y, alpha=0.3)
            
            ax4.set_xlabel('Error (m)')
            ax4.set_ylabel('Density')
            ax4.set_title('Error Distribution', fontweight='bold')
            ax4.legend()
            ax4.grid(alpha=0.3)
            
        # Metrics table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('tight')
        ax5.axis('off')
        
        # Prepare table data
        table_data = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'R2' in key:
                    formatted_value = f'{value:.4f}'
                elif '%' in key:
                    formatted_value = f'{value:.1f}%'
                else:
                    formatted_value = f'{value:.3f}'
            else:
                formatted_value = str(value)
            table_data.append([key, formatted_value])
            
        table = ax5.table(cellText=table_data, colLabels=['Metric', 'Value'],
                         cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)
        ax5.set_title('Complete Training Metrics', fontsize=12, fontweight='bold', pad=20)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    def _create_feature_importance_page(self, pdf: PdfPages):
        """Create feature importance analysis page."""
        
        fig = plt.figure(figsize=(8.5, 11))
        gs = GridSpec(2, 1, figure=fig, hspace=0.3)
        
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        if not self.results.feature_importance:
            ax = fig.add_subplot(gs[0, 0])
            ax.text(0.5, 0.5, 'No feature importance data available', 
                   ha='center', va='center', fontsize=16)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            return
            
        # Top features bar plot
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Sort features by importance
        sorted_features = sorted(self.results.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        # Take top 20 features
        top_features = sorted_features[:20]
        feature_names = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(feature_names))
        bars = ax1.barh(y_pos, importances, color=plt.cm.viridis(np.linspace(0, 1, len(importances))))
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(feature_names)
        ax1.invert_yaxis()  # Top feature at top
        ax1.set_xlabel('Importance Score')
        ax1.set_title('Top 20 Most Important Features', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importances)):
            width = bar.get_width()
            ax1.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', ha='left', va='center', fontsize=8)
                    
        # Feature importance distribution
        ax2 = fig.add_subplot(gs[1, 0])
        
        all_importances = list(self.results.feature_importance.values())
        non_zero_importances = [imp for imp in all_importances if imp > 0]
        
        ax2.hist(non_zero_importances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(non_zero_importances), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(non_zero_importances):.3f}')
        ax2.axvline(np.median(non_zero_importances), color='orange', linestyle='--',
                   label=f'Median: {np.median(non_zero_importances):.3f}')
        
        ax2.set_xlabel('Importance Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Feature Importance Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Statistics:
Total Features: {len(all_importances)}
Non-zero Features: {len(non_zero_importances)}
Zero Features: {len(all_importances) - len(non_zero_importances)}
Max Importance: {max(all_importances):.4f}
Min Importance: {min(non_zero_importances):.4f}"""
        
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9)
                
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    def _create_spatial_analysis_page(self, pdf: PdfPages, pred_analysis: Dict):
        """Create spatial analysis page."""
        
        fig = plt.figure(figsize=(8.5, 11))
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        fig.suptitle('Spatial Analysis', fontsize=16, fontweight='bold')
        
        if not pred_analysis['spatial_stats']:
            ax = fig.add_subplot(gs[1, :])
            ax.text(0.5, 0.5, 'No spatial analysis data available', 
                   ha='center', va='center', fontsize=16)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            return
            
        stats = pred_analysis['spatial_stats']
        
        # Coverage comparison
        ax1 = fig.add_subplot(gs[0, 0])
        files = list(stats.keys())
        coverages = [stats[f]['coverage_percent'] for f in files]
        
        bars = ax1.bar(range(len(files)), coverages, color='lightcoral')
        ax1.set_xticks(range(len(files)))
        ax1.set_xticklabels([f[:10] + '...' if len(f) > 10 else f for f in files], 
                           rotation=45, ha='right')
        ax1.set_ylabel('Coverage (%)')
        ax1.set_title('Prediction Coverage by File', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Height statistics comparison
        ax2 = fig.add_subplot(gs[0, 1])
        mean_heights = [stats[f]['mean_height'] for f in files]
        std_heights = [stats[f]['std_height'] for f in files]
        
        x = range(len(files))
        ax2.errorbar(x, mean_heights, yerr=std_heights, fmt='o-', capsize=5, capthick=2)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f[:10] + '...' if len(f) > 10 else f for f in files], 
                           rotation=45, ha='right')
        ax2.set_ylabel('Height (m)')
        ax2.set_title('Mean Height ± Std Dev', fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Height range comparison
        ax3 = fig.add_subplot(gs[1, :])
        
        if len(files) > 0:
            # Prepare data for grouped bar chart
            metrics = ['min_height', 'mean_height', 'max_height']
            x = np.arange(len(files))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                values = [stats[f][metric] for f in files]
                offset = (i - 1) * width
                label = metric.replace('_', ' ').title()
                ax3.bar(x + offset, values, width, label=label, alpha=0.8)
                
            ax3.set_xlabel('Prediction Files')
            ax3.set_ylabel('Height (m)')
            ax3.set_title('Height Range Comparison', fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels([f[:15] + '...' if len(f) > 15 else f for f in files], 
                               rotation=45, ha='right')
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
            
        # Summary statistics table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create summary table
        table_data = []
        for fname, fstats in stats.items():
            row = [
                fname[:20] + '...' if len(fname) > 20 else fname,
                f"{fstats['shape'][0]}×{fstats['shape'][1]}",
                f"{fstats['coverage_percent']:.1f}%",
                f"{fstats['mean_height']:.2f}m",
                f"{fstats['std_height']:.2f}m",
                f"{fstats['min_height']:.1f}-{fstats['max_height']:.1f}m"
            ]
            table_data.append(row)
            
        headers = ['File', 'Dimensions', 'Coverage', 'Mean Height', 'Std Height', 'Range']
        table = ax4.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        ax4.set_title('Spatial Statistics Summary', fontsize=12, fontweight='bold', pad=20)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    def _create_prediction_visualizations_page(self, pdf: PdfPages):
        """Create prediction visualizations page."""
        
        fig = plt.figure(figsize=(8.5, 11))
        
        if not self.results.prediction_files:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No prediction files available for visualization', 
                   ha='center', va='center', fontsize=16)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            return
            
        # Show up to 4 predictions
        n_preds = min(4, len(self.results.prediction_files))
        
        if n_preds == 1:
            gs = GridSpec(1, 1, figure=fig)
        elif n_preds == 2:
            gs = GridSpec(1, 2, figure=fig, wspace=0.3)
        else:
            gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
            
        fig.suptitle('Prediction Visualizations', fontsize=16, fontweight='bold')
        
        for i, pred_file in enumerate(self.results.prediction_files[:n_preds]):
            try:
                with rasterio.open(pred_file) as src:
                    prediction = src.read(1)
                    
                if n_preds == 1:
                    ax = fig.add_subplot(gs[0, 0])
                elif n_preds == 2:
                    ax = fig.add_subplot(gs[0, i])
                else:
                    row, col = i // 2, i % 2
                    ax = fig.add_subplot(gs[row, col])
                    
                # Create visualization
                valid_data = prediction[~np.isnan(prediction) & (prediction > 0)]
                if len(valid_data) > 0:
                    vmin, vmax = np.percentile(valid_data, [5, 95])
                    
                    im = ax.imshow(prediction, cmap='viridis', vmin=vmin, vmax=vmax)
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('Height (m)', rotation=270, labelpad=15)
                    
                    fname = Path(pred_file).name
                    title = fname[:25] + '...' if len(fname) > 25 else fname
                    ax.set_title(f'{title}\nMean: {valid_data.mean():.1f}m', fontsize=10)
                    ax.axis('off')
                else:
                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
                    ax.set_title(Path(pred_file).name, fontsize=10)
                    ax.axis('off')
                    
            except Exception as e:
                if n_preds == 1:
                    ax = fig.add_subplot(gs[0, 0])
                elif n_preds == 2:
                    ax = fig.add_subplot(gs[0, i])
                else:
                    row, col = i // 2, i % 2
                    ax = fig.add_subplot(gs[row, col])
                    
                ax.text(0.5, 0.5, f'Error loading:\n{str(e)[:50]}...', 
                       ha='center', va='center', fontsize=8)
                ax.set_title(Path(pred_file).name, fontsize=10)
                ax.axis('off')
                
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    def _create_multipatch_analysis_page(self, pdf: PdfPages):
        """Create multi-patch analysis page."""
        
        fig = plt.figure(figsize=(8.5, 11))
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        fig.suptitle('Multi-Patch Analysis', fontsize=16, fontweight='bold')
        
        if self.results.patch_summary is None:
            ax = fig.add_subplot(gs[1, :])
            ax.text(0.5, 0.5, 'No patch summary available', 
                   ha='center', va='center', fontsize=16)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            return
            
        df = self.results.patch_summary
        
        # Patch distribution map
        ax1 = fig.add_subplot(gs[0, :])
        
        if 'center_lon' in df.columns and 'center_lat' in df.columns:
            scatter = ax1.scatter(df['center_lon'], df['center_lat'], 
                                c=df.index, cmap='tab10', s=100, alpha=0.7)
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            ax1.set_title('Patch Spatial Distribution', fontweight='bold')
            ax1.grid(alpha=0.3)
            
            # Add patch labels
            for i, row in df.iterrows():
                ax1.annotate(f'P{i}', (row['center_lon'], row['center_lat']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
                           
        # Patch characteristics
        ax2 = fig.add_subplot(gs[1, 0])
        
        if 'band_count' in df.columns:
            band_counts = df['band_count'].value_counts()
            ax2.pie(band_counts.values, labels=[f'{bc} bands' for bc in band_counts.index], 
                   autopct='%1.1f%%')
            ax2.set_title('Band Count Distribution', fontweight='bold')
            
        # Temporal mode distribution
        ax3 = fig.add_subplot(gs[1, 1])
        
        if 'temporal_mode' in df.columns:
            temporal_counts = df['temporal_mode'].value_counts()
            colors = ['lightcoral', 'lightblue']
            ax3.pie(temporal_counts.values, 
                   labels=['Temporal' if tm else 'Non-temporal' for tm in temporal_counts.index],
                   autopct='%1.1f%%', colors=colors)
            ax3.set_title('Processing Mode Distribution', fontweight='bold')
            
        # Patch summary table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('tight')
        ax4.axis('off')
        
        # Select key columns for display
        display_cols = ['patch_id', 'temporal_mode', 'band_count', 'width', 'height']
        if 'pixel_size_m' in df.columns:
            display_cols.append('pixel_size_m')
            
        display_df = df[display_cols].head(10)  # Show first 10 patches
        
        table_data = display_df.values.tolist()
        headers = [col.replace('_', ' ').title() for col in display_cols]
        
        table = ax4.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        title = f'Patch Summary (showing {len(display_df)} of {len(df)} patches)'
        ax4.set_title(title, fontsize=12, fontweight='bold', pad=20)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(
        description='Generate comprehensive evaluation reports for train_predict_map.py outputs'
    )
    
    parser.add_argument('output_dir', type=str, nargs='?', default='.',
                       help='Directory containing train_predict_map.py outputs')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output PDF file path (default: auto-generated)')
    parser.add_argument('--legacy', action='store_true',
                       help='Run legacy temporal evaluation for backwards compatibility')
    
    args = parser.parse_args()
    
    if args.legacy:
        # Legacy mode for backwards compatibility
        evaluate_temporal_prediction()
        check_training_progress()
        return
        
    try:
        # Create evaluation system
        evaluator = UnifiedEvaluationSystem(args.output_dir)
        
        # Load results
        results = evaluator.load_training_results()
        
        # Generate PDF report
        pdf_path = evaluator.generate_pdf_report(args.output)
        
        print(f"\n✅ Evaluation complete!")
        print(f"📄 Report saved: {pdf_path}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

# Legacy functions for backwards compatibility
def evaluate_temporal_prediction():
    """Legacy function for backwards compatibility."""
    
    print("🔍 Running legacy temporal prediction evaluation...")
    
    # Check if prediction exists
    pred_path = "chm_outputs/improved_temporal_prediction.tif"
    if not os.path.exists(pred_path):
        print(f"❌ Prediction file not found: {pred_path}")
        return
    
    # Load prediction
    with rasterio.open(pred_path) as src:
        prediction = src.read(1)
    
    # Calculate basic statistics
    pred_valid = prediction[~np.isnan(prediction) & (prediction > 0)]
    
    print(f"\nPrediction statistics:")
    print(f"  Range: {pred_valid.min():.2f}m to {pred_valid.max():.2f}m")
    print(f"  Mean: {pred_valid.mean():.2f}m")
    print(f"  Valid pixels: {len(pred_valid)}/{prediction.size}")
    
    # Create simple visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    im = ax.imshow(prediction, cmap='viridis', vmin=0, vmax=50)
    ax.set_title(f'Temporal 3D U-Net Prediction\nMean: {pred_valid.mean():.1f}m')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Height (m)')
    
    plt.tight_layout()
    plt.savefig('chm_outputs/temporal_evaluation.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Evaluation plot saved: chm_outputs/temporal_evaluation.png")

def check_training_progress():
    """Legacy function for backwards compatibility."""
    
    print("\n📊 Checking training progress...")
    
    # Check for loss plot
    loss_plot = "chm_outputs/improved_temporal_loss.png"
    if os.path.exists(loss_plot):
        print(f"✅ Loss plot available: {loss_plot}")
    
    # Check for checkpoints
    checkpoints = list(Path("chm_outputs").glob("improved_temporal_epoch_*.pth"))
    if checkpoints:
        print(f"✅ Training checkpoints: {len(checkpoints)}")
        for cp in sorted(checkpoints):
            epoch = cp.stem.split('_')[-1]
            print(f"   Epoch {epoch}: {cp}")
    
    # Check for final model
    final_model = "chm_outputs/improved_temporal_final.pth"
    if os.path.exists(final_model):
        print(f"✅ Final model: {final_model}")
    else:
        print(f"⏳ Final model not yet available")

if __name__ == "__main__":
    exit(main())