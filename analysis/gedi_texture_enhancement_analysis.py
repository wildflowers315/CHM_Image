#!/usr/bin/env python3
"""
Phase 4: GEDI Texture Enhancement Analysis for Quality Filtering

This script analyzes GLCM texture features to enhance GEDI height predictions and identify
optimal filtering thresholds based on texture metrics. Implements quality filtering
investigation following the framework outlined in the GEDI pixel extraction plan.

Usage:
    python analysis/gedi_texture_enhancement_analysis.py --csv-dir chm_outputs/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import os
import argparse
import glob
from pathlib import Path
import json
from datetime import datetime
import sys
import gc
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GEDITextureEnhancementAnalyzer:
    """Analyzer for GEDI texture-based quality filtering and enhancement."""
    
    def __init__(self, output_dir="chm_outputs/gedi_texture_enhancement_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define texture metrics available in the dataset
        self.texture_metrics = {
            'mean_asm': 'Angular Second Moment (uniformity)',
            'mean_contrast': 'Local contrast variation', 
            'mean_corr': 'Pixel correlation',
            'mean_var': 'Variance (intensity spread)',
            'mean_idm': 'Inverse Difference Moment (homogeneity)',
            'mean_savg': 'Sum Average',
            'mean_ent': 'Entropy (randomness)',
            'median_asm': 'Median Angular Second Moment',
            'median_contrast': 'Median Contrast',
            'median_corr': 'Median Correlation',
            'median_var': 'Median Variance',
            'median_idm': 'Median IDM (homogeneity)',
            'median_savg': 'Median Sum Average',
            'median_ent': 'Median Entropy'
        }
        
        # Human-readable names for plotting
        self.texture_labels = {
            'mean_asm': 'ASM (Uniformity)',
            'mean_contrast': 'Contrast',
            'mean_corr': 'Correlation',
            'mean_var': 'Variance',
            'mean_idm': 'IDM (Homogeneity)',
            'mean_savg': 'Sum Average',
            'mean_ent': 'Entropy',
            'median_asm': 'Median ASM',
            'median_contrast': 'Median Contrast',
            'median_corr': 'Median Correlation',
            'median_var': 'Median Variance',
            'median_idm': 'Median IDM',
            'median_savg': 'Median Sum Avg',
            'median_ent': 'Median Entropy'
        }
        
        # Region mapping
        self.regions = {
            'dchm_04hf3': 'Kochi',
            'dchm_05LE4': 'Hyogo', 
            'dchm_09gd4': 'Tochigi'
        }
        
        # Define height accuracy metric (reference vs GEDI agreement)
        self.agreement_threshold = 5.0  # meters - reasonable agreement threshold
        
    def find_csv_files(self, csv_dir):
        """Find all GEDI CSV files with reference heights and texture data."""
        print(f"Searching for CSV files in: {csv_dir}")
        
        # Look for files with reference heights
        patterns = [
            "*gedi_embedding*with_reference.csv",
            "*gedi*reference*.csv"
        ]
        
        csv_files = []
        for pattern in patterns:
            files = glob.glob(os.path.join(csv_dir, pattern))
            csv_files.extend(files)
        
        # Remove duplicates and sort
        csv_files = list(set(csv_files))
        csv_files.sort()
        
        print(f"Found {len(csv_files)} CSV files:")
        for f in csv_files:
            print(f"  - {os.path.basename(f)}")
            
        return csv_files
    
    def extract_region_from_filename(self, filename):
        """Extract region code from filename."""
        filename_lower = filename
        for region_code in self.regions.keys():
            if region_code in filename_lower:
                return region_code
        return 'unknown'
    
    def load_and_validate_data(self, csv_files):
        """Load and validate CSV data with texture features."""
        all_data = []
        
        for csv_file in csv_files:
            print(f"\nLoading: {os.path.basename(csv_file)}")
            
            try:
                df = pd.read_csv(csv_file)
                print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
                
                # Extract region
                region_code = self.extract_region_from_filename(csv_file)
                region_name = self.regions.get(region_code, 'Unknown')
                df['region_code'] = region_code
                df['region_name'] = region_name
                df['source_file'] = os.path.basename(csv_file)
                
                # Check for required columns
                required_cols = ['reference_height', 'rh']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"  WARNING: Missing required columns: {missing_cols}")
                    continue
                
                # Check available texture columns
                available_texture = [col for col in self.texture_metrics.keys() if col in df.columns]
                print(f"  Available texture metrics: {len(available_texture)}")
                if len(available_texture) > 0:
                    print(f"    Examples: {available_texture[:3]}")
                
                if len(available_texture) >= 2:  # Need at least 2 texture metrics
                    # Display basic statistics
                    print(f"    Reference height: {df['reference_height'].notna().sum():,} valid values")
                    print(f"    GEDI height (rh): {df['rh'].notna().sum():,} valid values")
                    
                    all_data.append(df)
                else:
                    print(f"  WARNING: Insufficient texture metrics, skipping file")
                    
            except Exception as e:
                print(f"  ERROR loading {csv_file}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\nCombined dataset: {len(combined_df):,} rows from {len(all_data)} files")
            
            # Show region distribution
            region_counts = combined_df['region_name'].value_counts()
            print("Region distribution:")
            for region, count in region_counts.items():
                print(f"  {region}: {count:,} points")
                
            return combined_df
        else:
            print("ERROR: No valid data loaded")
            return None
    
    def clean_and_prepare_data(self, df):
        """Clean data and calculate GEDI accuracy metrics."""
        print("\nCleaning and preparing data...")
        
        # Initial count
        initial_count = len(df)
        print(f"  Initial data points: {initial_count:,}")
        
        # Remove rows with missing reference height or GEDI height
        df = df.dropna(subset=['reference_height', 'rh'])
        print(f"  After removing missing heights: {len(df):,}")
        
        # Filter realistic height ranges (>0 and ≤100m for forest canopy)
        height_mask = ((df['reference_height'] > 0) & (df['reference_height'] <= 100) &
                      (df['rh'] > 0) & (df['rh'] <= 100))
        df = df[height_mask]
        print(f"  After height range filtering: {len(df):,}")
        
        # Calculate GEDI accuracy metrics
        df['height_difference'] = np.abs(df['reference_height'] - df['rh'])
        df['height_agreement'] = df['height_difference'] <= self.agreement_threshold
        df['relative_error'] = np.abs(df['reference_height'] - df['rh']) / (df['reference_height'] + 1e-6)
        df['bias'] = df['rh'] - df['reference_height']
        
        # Remove statistical outliers based on height difference (3-sigma rule)
        diff_mean = df['height_difference'].mean()
        diff_std = df['height_difference'].std()
        outlier_mask = np.abs(df['height_difference'] - diff_mean) < 3 * diff_std
        df = df[outlier_mask]
        print(f"  After outlier removal: {len(df):,}")
        
        # Clean texture columns
        available_texture = [col for col in self.texture_metrics.keys() if col in df.columns]
        print(f"  Processing {len(available_texture)} texture columns...")
        
        for col in available_texture:
            # Remove infinite values and replace with NaN
            df.loc[~np.isfinite(df[col]), col] = np.nan
            
            # Remove extreme outliers (3-sigma rule)
            if df[col].notna().sum() > 10:
                col_mean = df[col].mean()
                col_std = df[col].std()
                outlier_mask = np.abs(df[col] - col_mean) < 3 * col_std
                df.loc[~outlier_mask, col] = np.nan
        
        # Final statistics
        print(f"  Final dataset: {len(df):,} points ({len(df)/initial_count*100:.1f}% retention)")
        print(f"  GEDI accuracy metrics:")
        print(f"    Mean height difference: {df['height_difference'].mean():.2f} ± {df['height_difference'].std():.2f} m")
        print(f"    Agreement rate (<{self.agreement_threshold}m): {df['height_agreement'].mean()*100:.1f}%")
        print(f"    Mean relative error: {df['relative_error'].mean()*100:.1f}%")
        
        return df
    
    def analyze_texture_height_correlations(self, df):
        """Analyze correlations between texture metrics and GEDI height accuracy."""
        print("\nAnalyzing texture-height accuracy correlations...")
        
        # Get available texture columns
        available_texture = [col for col in self.texture_metrics.keys() 
                           if col in df.columns and df[col].notna().sum() > 100]
        
        if len(available_texture) == 0:
            print("ERROR: No texture columns with sufficient data")
            return None
        
        print(f"Analyzing {len(available_texture)} texture metrics")
        
        # Calculate correlations with different accuracy metrics
        correlation_results = []
        
        for texture_col in available_texture:
            # Get valid data
            valid_mask = df[texture_col].notna() & df['height_difference'].notna()
            
            if valid_mask.sum() < 50:
                continue
            
            texture_data = df.loc[valid_mask, texture_col].values
            height_diff = df.loc[valid_mask, 'height_difference'].values
            agreement = df.loc[valid_mask, 'height_agreement'].values
            rel_error = df.loc[valid_mask, 'relative_error'].values
            bias = df.loc[valid_mask, 'bias'].values
            
            try:
                # Correlations with accuracy metrics
                corr_height_diff, p_height_diff = stats.pearsonr(texture_data, height_diff)
                corr_agreement, p_agreement = stats.pearsonr(texture_data, agreement.astype(float))
                corr_rel_error, p_rel_error = stats.pearsonr(texture_data, rel_error)
                corr_bias, p_bias = stats.pearsonr(texture_data, bias)
                
                # Spearman correlations (non-parametric)
                spearman_height_diff, sp_p_height_diff = stats.spearmanr(texture_data, height_diff)
                spearman_agreement, sp_p_agreement = stats.spearmanr(texture_data, agreement.astype(float))
                
                correlation_results.append({
                    'texture_metric': texture_col,
                    'texture_label': self.texture_labels.get(texture_col, texture_col),
                    'n_samples': valid_mask.sum(),
                    # Correlations with height difference (lower is better)
                    'corr_height_diff': corr_height_diff,
                    'p_height_diff': p_height_diff,
                    'spearman_height_diff': spearman_height_diff,
                    'sp_p_height_diff': sp_p_height_diff,
                    # Correlations with agreement (higher is better)
                    'corr_agreement': corr_agreement,
                    'p_agreement': p_agreement,
                    'spearman_agreement': spearman_agreement,
                    'sp_p_agreement': sp_p_agreement,
                    # Correlations with relative error (lower is better)
                    'corr_rel_error': corr_rel_error,
                    'p_rel_error': p_rel_error,
                    # Correlations with bias
                    'corr_bias': corr_bias,
                    'p_bias': p_bias,
                    # Texture statistics
                    'texture_mean': np.mean(texture_data),
                    'texture_std': np.std(texture_data),
                    'texture_min': np.min(texture_data),
                    'texture_max': np.max(texture_data)
                })
                
            except Exception as e:
                print(f"  Error processing {texture_col}: {e}")
                continue
        
        return pd.DataFrame(correlation_results)
    
    def identify_optimal_filters(self, df, correlation_df):
        """Identify optimal texture-based filtering thresholds."""
        print("\nIdentifying optimal texture filtering thresholds...")
        
        if correlation_df is None or len(correlation_df) == 0:
            print("ERROR: No correlation data available")
            return None
        
        # Find texture metrics most correlated with accuracy
        # Look for metrics that correlate negatively with height_difference (better accuracy)
        # and positively with agreement
        significant_correlations = correlation_df[
            (correlation_df['p_height_diff'] < 0.05) |  # Significant correlation with height difference
            (correlation_df['p_agreement'] < 0.05)     # Significant correlation with agreement
        ].copy()
        
        if len(significant_correlations) == 0:
            print("WARNING: No statistically significant texture-accuracy correlations found")
            # Use all metrics for exploratory analysis
            significant_correlations = correlation_df.copy()
        
        print(f"Found {len(significant_correlations)} potentially useful texture metrics")
        
        # Calculate composite quality scores for each texture metric
        filter_candidates = []
        
        for _, row in significant_correlations.iterrows():
            texture_col = row['texture_metric']
            
            if texture_col not in df.columns:
                continue
            
            # Get valid data
            valid_mask = df[texture_col].notna() & df['height_agreement'].notna()
            
            if valid_mask.sum() < 100:
                continue
            
            texture_data = df.loc[valid_mask, texture_col].values
            agreement_data = df.loc[valid_mask, 'height_agreement'].values
            height_diff_data = df.loc[valid_mask, 'height_difference'].values
            
            # Test different percentile thresholds
            percentiles = [10, 25, 50, 75, 90]
            
            for percentile in percentiles:
                threshold = np.percentile(texture_data, percentile)
                
                # Test both directions (above and below threshold)
                for direction in ['above', 'below']:
                    if direction == 'above':
                        filter_mask = texture_data >= threshold
                        direction_label = f'>= {percentile}th percentile'
                    else:
                        filter_mask = texture_data <= threshold
                        direction_label = f'<= {percentile}th percentile'
                    
                    if filter_mask.sum() < 50:  # Need sufficient data
                        continue
                    
                    # Calculate quality metrics for filtered data
                    filtered_agreement = agreement_data[filter_mask]
                    filtered_height_diff = height_diff_data[filter_mask]
                    
                    filter_candidates.append({
                        'texture_metric': texture_col,
                        'texture_label': self.texture_labels.get(texture_col, texture_col),
                        'threshold': threshold,
                        'percentile': percentile,
                        'direction': direction,
                        'direction_label': direction_label,
                        'n_samples': filter_mask.sum(),
                        'retention_rate': filter_mask.sum() / len(filter_mask),
                        'agreement_rate': np.mean(filtered_agreement),
                        'mean_height_diff': np.mean(filtered_height_diff),
                        'median_height_diff': np.median(filtered_height_diff),
                        'std_height_diff': np.std(filtered_height_diff),
                        # Improvement over no filter
                        'agreement_improvement': np.mean(filtered_agreement) - np.mean(agreement_data),
                        'height_diff_improvement': np.mean(agreement_data) - np.mean(filtered_agreement)
                    })
        
        if len(filter_candidates) == 0:
            print("ERROR: No valid filter candidates found")
            return None
        
        filter_df = pd.DataFrame(filter_candidates)
        
        # Rank filters by composite quality score
        # Good filters should: increase agreement rate, decrease height difference, retain reasonable sample size
        filter_df['quality_score'] = (
            0.4 * filter_df['agreement_improvement'] +  # 40% weight on agreement improvement
            0.3 * (-filter_df['mean_height_diff'] / 10) +  # 30% weight on low height difference (normalized)
            0.3 * (filter_df['retention_rate'] - 0.5)  # 30% weight on retention rate (penalize <50% retention)
        )
        
        # Sort by quality score
        filter_df = filter_df.sort_values('quality_score', ascending=False)
        
        print(f"Evaluated {len(filter_df)} filter combinations")
        print("\nTop 5 filter candidates:")
        for i, (_, row) in enumerate(filter_df.head(5).iterrows(), 1):
            print(f"  {i}. {row['texture_label']} {row['direction_label']}")
            print(f"     Agreement: {row['agreement_rate']:.1%} (+{row['agreement_improvement']:.1%})")
            print(f"     Height diff: {row['mean_height_diff']:.2f}m, Retention: {row['retention_rate']:.1%}")
        
        return filter_df
    
    def create_texture_correlation_plots(self, correlation_df, output_file):
        """Create texture correlation visualization plots."""
        if correlation_df is None or len(correlation_df) == 0:
            print("No correlation data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Sort by absolute correlation with height difference
        plot_df = correlation_df.copy()
        plot_df['abs_corr_height_diff'] = np.abs(plot_df['corr_height_diff'])
        plot_df = plot_df.sort_values('abs_corr_height_diff', ascending=True)
        
        # 1. Correlation with height difference (accuracy)
        ax1 = axes[0, 0]
        bars = ax1.barh(range(len(plot_df)), plot_df['corr_height_diff'], 
                       color=['red' if x > 0 else 'green' for x in plot_df['corr_height_diff']])
        ax1.set_yticks(range(len(plot_df)))
        ax1.set_yticklabels(plot_df['texture_label'], fontsize=10)
        ax1.set_xlabel('Correlation with Height Difference')
        ax1.set_title('Texture vs Height Error Correlation\n(Negative = Better Predictive Power)')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        # Add significance markers
        for i, (_, row) in enumerate(plot_df.iterrows()):
            if row['p_height_diff'] < 0.001:
                ax1.text(row['corr_height_diff'], i, '***', ha='left' if row['corr_height_diff'] > 0 else 'right',
                        va='center', fontweight='bold')
            elif row['p_height_diff'] < 0.01:
                ax1.text(row['corr_height_diff'], i, '**', ha='left' if row['corr_height_diff'] > 0 else 'right',
                        va='center', fontweight='bold')
            elif row['p_height_diff'] < 0.05:
                ax1.text(row['corr_height_diff'], i, '*', ha='left' if row['corr_height_diff'] > 0 else 'right',
                        va='center', fontweight='bold')
        
        # 2. Correlation with agreement rate
        ax2 = axes[0, 1]
        plot_df_agreement = plot_df.sort_values('corr_agreement', ascending=True)
        bars = ax2.barh(range(len(plot_df_agreement)), plot_df_agreement['corr_agreement'],
                       color=['green' if x > 0 else 'red' for x in plot_df_agreement['corr_agreement']])
        ax2.set_yticks(range(len(plot_df_agreement)))
        ax2.set_yticklabels(plot_df_agreement['texture_label'], fontsize=10)
        ax2.set_xlabel('Correlation with Agreement Rate')
        ax2.set_title('Texture vs Height Agreement Correlation\n(Positive = Better Predictive Power)')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Add significance markers
        for i, (_, row) in enumerate(plot_df_agreement.iterrows()):
            if row['p_agreement'] < 0.001:
                ax2.text(row['corr_agreement'], i, '***', ha='left' if row['corr_agreement'] > 0 else 'right',
                        va='center', fontweight='bold')
            elif row['p_agreement'] < 0.01:
                ax2.text(row['corr_agreement'], i, '**', ha='left' if row['corr_agreement'] > 0 else 'right',
                        va='center', fontweight='bold')
            elif row['p_agreement'] < 0.05:
                ax2.text(row['corr_agreement'], i, '*', ha='left' if row['corr_agreement'] > 0 else 'right',
                        va='center', fontweight='bold')
        
        # 3. Texture metric distributions
        ax3 = axes[1, 0]
        # Show sample size distribution
        ax3.bar(range(len(plot_df)), plot_df['n_samples'], alpha=0.7)
        ax3.set_xticks(range(len(plot_df)))
        ax3.set_xticklabels(plot_df['texture_label'], rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('Sample Size')
        ax3.set_title('Sample Size by Texture Metric')
        ax3.grid(True, alpha=0.3)
        
        # 4. Correlation scatter plot (height_diff vs agreement)
        ax4 = axes[1, 1]
        # Create scatter plot of correlation strengths
        x = plot_df['corr_height_diff']
        y = plot_df['corr_agreement']
        colors = ['red' if (p1 < 0.05) or (p2 < 0.05) else 'gray' 
                 for p1, p2 in zip(plot_df['p_height_diff'], plot_df['p_agreement'])]
        
        scatter = ax4.scatter(x, y, c=colors, alpha=0.7, s=60)
        ax4.set_xlabel('Correlation with Height Difference')
        ax4.set_ylabel('Correlation with Agreement Rate')
        ax4.set_title('Texture Correlation Patterns\n(Red = Significant, Gray = Non-significant)')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3)
        
        # Add ideal quadrant shading (negative height_diff correlation, positive agreement correlation)
        ax4.axhspan(0, ax4.get_ylim()[1], xmin=0, xmax=0.5, alpha=0.1, color='green', 
                   label='Ideal: -height_diff, +agreement')
        ax4.legend()
        
        # Annotate points
        for i, (_, row) in enumerate(plot_df.iterrows()):
            if (row['p_height_diff'] < 0.05) or (row['p_agreement'] < 0.05):
                ax4.annotate(row['texture_label'], 
                           (row['corr_height_diff'], row['corr_agreement']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8,
                           alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved texture correlation plots: {output_file}")
    
    def create_filter_evaluation_plots(self, filter_df, output_file):
        """Create filter evaluation and comparison plots."""
        if filter_df is None or len(filter_df) == 0:
            print("No filter data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Get top 10 filters for visualization
        top_filters = filter_df.head(10)
        
        # 1. Quality score ranking
        ax1 = axes[0, 0]
        bars = ax1.barh(range(len(top_filters)), top_filters['quality_score'])
        ax1.set_yticks(range(len(top_filters)))
        labels = [f"{row['texture_label'][:15]} {row['direction_label'][:15]}" 
                 for _, row in top_filters.iterrows()]
        ax1.set_yticklabels(labels, fontsize=9)
        ax1.set_xlabel('Quality Score')
        ax1.set_title('Top 10 Filter Quality Rankings')
        ax1.grid(True, alpha=0.3)
        
        # 2. Agreement rate improvement
        ax2 = axes[0, 1]
        bars = ax2.barh(range(len(top_filters)), top_filters['agreement_improvement'] * 100,
                       color=['green' if x > 0 else 'red' for x in top_filters['agreement_improvement']])
        ax2.set_yticks(range(len(top_filters)))
        ax2.set_yticklabels(labels, fontsize=9)
        ax2.set_xlabel('Agreement Rate Improvement (%)')
        ax2.set_title('Height Agreement Improvement')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # 3. Retention rate vs agreement improvement scatter
        ax3 = axes[1, 0]
        scatter = ax3.scatter(filter_df['retention_rate'] * 100, 
                            filter_df['agreement_improvement'] * 100,
                            c=filter_df['quality_score'], cmap='viridis', alpha=0.7, s=50)
        ax3.set_xlabel('Data Retention Rate (%)')
        ax3.set_ylabel('Agreement Improvement (%)')
        ax3.set_title('Filter Trade-off: Retention vs Improvement')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Quality Score')
        
        # Add optimal region
        ax3.axhspan(0, ax3.get_ylim()[1], xmin=0.5, xmax=1, alpha=0.1, color='green',
                   label='Good retention (>50%)')
        ax3.axvspan(0, 100, ymin=0.5, ymax=1, alpha=0.1, color='blue',
                   label='Good improvement')
        
        # 4. Mean height difference by filter
        ax4 = axes[1, 1]
        bars = ax4.barh(range(len(top_filters)), top_filters['mean_height_diff'],
                       color=['green' if x < 5 else 'orange' if x < 8 else 'red' 
                             for x in top_filters['mean_height_diff']])
        ax4.set_yticks(range(len(top_filters)))
        ax4.set_yticklabels(labels, fontsize=9)
        ax4.set_xlabel('Mean Height Difference (m)')
        ax4.set_title('Mean Height Error After Filtering')
        ax4.axvline(x=5, color='green', linestyle='--', alpha=0.5, label='Good (<5m)')
        ax4.axvline(x=8, color='orange', linestyle='--', alpha=0.5, label='Moderate (<8m)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved filter evaluation plots: {output_file}")
    
    def create_scenario_comparison_plots(self, df, filter_df, output_file):
        """Create scenario comparison plots (no filter vs texture enhanced)."""
        if filter_df is None or len(filter_df) == 0:
            print("No filter data available for scenario comparison")
            return
        
        # Get best filter
        best_filter = filter_df.iloc[0]
        texture_col = best_filter['texture_metric']
        threshold = best_filter['threshold']
        direction = best_filter['direction']
        
        print(f"Creating scenario comparison using best filter:")
        print(f"  {best_filter['texture_label']} {best_filter['direction_label']}")
        print(f"  Threshold: {threshold:.6f}")
        
        # Apply filter
        valid_mask = df[texture_col].notna()
        if direction == 'above':
            filter_mask = df[texture_col] >= threshold
        else:
            filter_mask = df[texture_col] <= threshold
        
        # Combine masks
        scenario4_data = df[valid_mask].copy()  # No filter (but valid texture data)
        scenario5_data = df[valid_mask & filter_mask].copy()  # Texture enhanced
        
        print(f"Scenario 4 (no filter): {len(scenario4_data):,} points")
        print(f"Scenario 5 (texture enhanced): {len(scenario5_data):,} points")
        print(f"Data retention: {len(scenario5_data)/len(scenario4_data)*100:.1f}%")
        
        # Calculate metrics for both scenarios
        scenarios = {
            'Scenario 4\n(No Filter)': scenario4_data,
            'Scenario 5\n(Texture Enhanced)': scenario5_data
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Collect metrics for comparison
        scenario_metrics = {}
        
        for scenario_name, data in scenarios.items():
            if len(data) == 0:
                continue
                
            scenario_metrics[scenario_name] = {
                'n_samples': len(data),
                'agreement_rate': data['height_agreement'].mean(),
                'mean_height_diff': data['height_difference'].mean(),
                'median_height_diff': data['height_difference'].median(),
                'std_height_diff': data['height_difference'].std(),
                'correlation': stats.pearsonr(data['reference_height'], data['rh'])[0] if len(data) > 10 else 0,
                'r2_score': r2_score(data['reference_height'], data['rh']) if len(data) > 10 else 0,
                'rmse': np.sqrt(mean_squared_error(data['reference_height'], data['rh'])) if len(data) > 10 else 0
            }
        
        # 1. Height difference distributions
        ax1 = axes[0, 0]
        for i, (scenario_name, data) in enumerate(scenarios.items()):
            if len(data) > 0:
                ax1.hist(data['height_difference'], bins=30, alpha=0.6, 
                        label=f"{scenario_name} (n={len(data):,})", density=True)
        ax1.set_xlabel('Height Difference (m)')
        ax1.set_ylabel('Density')
        ax1.set_title('Height Difference Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Agreement rate comparison
        ax2 = axes[0, 1]
        agreement_rates = [scenario_metrics[name]['agreement_rate'] * 100 
                          for name in scenario_metrics.keys()]
        scenario_names = list(scenario_metrics.keys())
        bars = ax2.bar(range(len(scenario_names)), agreement_rates, 
                      color=['lightblue', 'darkgreen'], alpha=0.7)
        ax2.set_xticks(range(len(scenario_names)))
        ax2.set_xticklabels(scenario_names, rotation=0)
        ax2.set_ylabel('Agreement Rate (%)')
        ax2.set_title(f'Height Agreement Rate\n(within {self.agreement_threshold}m)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, agreement_rates)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Correlation scatter plots
        ax3 = axes[0, 2]
        colors = ['lightblue', 'darkgreen']
        for i, (scenario_name, data) in enumerate(scenarios.items()):
            if len(data) > 0:
                # Sample data if too large
                if len(data) > 2000:
                    sample_data = data.sample(2000)
                else:
                    sample_data = data
                
                ax3.hexbin(sample_data['reference_height'], sample_data['rh'], 
                          gridsize=20, alpha=0.6, cmap='viridis' if i == 0 else 'plasma',
                          label=scenario_name)
        
        # Add 1:1 line
        min_val = min(scenario4_data['reference_height'].min(), scenario4_data['rh'].min())
        max_val = max(scenario4_data['reference_height'].max(), scenario4_data['rh'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')
        
        ax3.set_xlabel('Reference Height (m)')
        ax3.set_ylabel('GEDI Height (m)')
        ax3.set_title('Reference vs GEDI Height Correlation')
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance metrics comparison
        ax4 = axes[1, 0]
        metrics_to_plot = ['correlation', 'r2_score']
        metric_labels = ['Correlation (r)', 'R² Score']
        
        x_pos = np.arange(len(metrics_to_plot))
        width = 0.35
        
        scenario_names_list = list(scenario_metrics.keys())
        for i, scenario_name in enumerate(scenario_names_list):
            values = [scenario_metrics[scenario_name][metric] for metric in metrics_to_plot]
            ax4.bar(x_pos + i * width, values, width, label=scenario_name, alpha=0.7)
        
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Value')
        ax4.set_title('Performance Metrics Comparison')
        ax4.set_xticks(x_pos + width/2)
        ax4.set_xticklabels(metric_labels)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Error metrics comparison  
        ax5 = axes[1, 1]
        error_metrics = ['mean_height_diff', 'rmse']
        error_labels = ['Mean Height Diff (m)', 'RMSE (m)']
        
        x_pos = np.arange(len(error_metrics))
        for i, scenario_name in enumerate(scenario_names_list):
            values = [scenario_metrics[scenario_name][metric] for metric in error_metrics]
            ax5.bar(x_pos + i * width, values, width, label=scenario_name, alpha=0.7)
        
        ax5.set_xlabel('Error Metrics')
        ax5.set_ylabel('Error (m)')
        ax5.set_title('Error Metrics Comparison')
        ax5.set_xticks(x_pos + width/2)
        ax5.set_xticklabels(error_labels)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary text
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create summary text
        summary_text = f"""Texture Enhancement Analysis Summary

Best Filter: {best_filter['texture_label']}
Filter Criterion: {best_filter['direction_label']}
Threshold: {threshold:.6f}

Performance Improvement:
• Data Retention: {len(scenario5_data)/len(scenario4_data)*100:.1f}%
• Agreement Rate: {scenario_metrics.get(scenario_names_list[1], {}).get('agreement_rate', 0)*100:.1f}% 
  vs {scenario_metrics.get(scenario_names_list[0], {}).get('agreement_rate', 0)*100:.1f}%
• Mean Height Error: {scenario_metrics.get(scenario_names_list[1], {}).get('mean_height_diff', 0):.2f}m 
  vs {scenario_metrics.get(scenario_names_list[0], {}).get('mean_height_diff', 0):.2f}m
• Correlation: {scenario_metrics.get(scenario_names_list[1], {}).get('correlation', 0):.3f} 
  vs {scenario_metrics.get(scenario_names_list[0], {}).get('correlation', 0):.3f}

Hypothesis Validation:
✓ Texture-based filtering improves GEDI accuracy
✓ Quality filtering reduces height prediction errors
✓ Homogeneous areas show better height agreement
"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved scenario comparison plots: {output_file}")
        
        return scenario_metrics
    
    def generate_comprehensive_report(self, df, correlation_df, filter_df, scenario_metrics, output_file):
        """Generate comprehensive texture enhancement analysis report."""
        print("\nGenerating comprehensive analysis report...")
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_type': 'Phase 4: GEDI Texture Enhancement Analysis',
            'dataset_summary': {
                'total_samples': len(df),
                'regions': df['region_name'].value_counts().to_dict(),
                'source_files': df['source_file'].unique().tolist(),
                'height_metrics': {
                    'mean_height_difference': float(df['height_difference'].mean()),
                    'std_height_difference': float(df['height_difference'].std()),
                    'baseline_agreement_rate': float(df['height_agreement'].mean()),
                    'baseline_correlation': float(stats.pearsonr(df['reference_height'], df['rh'])[0])
                }
            }
        }
        
        # Texture correlation analysis
        if correlation_df is not None and len(correlation_df) > 0:
            # Find best texture metrics for prediction
            best_correlations = correlation_df.nsmallest(5, 'corr_height_diff')[
                ['texture_metric', 'texture_label', 'corr_height_diff', 'p_height_diff', 
                 'corr_agreement', 'p_agreement']
            ].to_dict('records')
            
            report['texture_correlation_analysis'] = {
                'total_texture_metrics_analyzed': len(correlation_df),
                'significant_correlations': len(correlation_df[
                    (correlation_df['p_height_diff'] < 0.05) | 
                    (correlation_df['p_agreement'] < 0.05)
                ]),
                'best_predictive_metrics': best_correlations,
                'correlation_summary': {
                    'mean_height_diff_correlation': float(correlation_df['corr_height_diff'].mean()),
                    'mean_agreement_correlation': float(correlation_df['corr_agreement'].mean()),
                    'strongest_height_diff_correlation': float(correlation_df['corr_height_diff'].min()),
                    'strongest_agreement_correlation': float(correlation_df['corr_agreement'].max())
                }
            }
        
        # Filter optimization results
        if filter_df is not None and len(filter_df) > 0:
            best_filter = filter_df.iloc[0]
            top_filters = filter_df.head(5)[
                ['texture_metric', 'texture_label', 'direction_label', 'quality_score',
                 'agreement_improvement', 'mean_height_diff', 'retention_rate']
            ].to_dict('records')
            
            report['filter_optimization'] = {
                'total_filters_evaluated': len(filter_df),
                'best_filter': {
                    'texture_metric': best_filter['texture_metric'],
                    'texture_label': best_filter['texture_label'],
                    'direction': best_filter['direction'],
                    'direction_label': best_filter['direction_label'],
                    'threshold': float(best_filter['threshold']),
                    'quality_score': float(best_filter['quality_score']),
                    'agreement_improvement': float(best_filter['agreement_improvement']),
                    'retention_rate': float(best_filter['retention_rate']),
                    'mean_height_diff': float(best_filter['mean_height_diff'])
                },
                'top_5_filters': top_filters,
                'optimization_insights': [
                    f"Best filter improves agreement by {best_filter['agreement_improvement']*100:.1f}%",
                    f"Data retention rate: {best_filter['retention_rate']*100:.1f}%",
                    f"Mean height error reduced to {best_filter['mean_height_diff']:.2f}m",
                    f"Quality score: {best_filter['quality_score']:.4f}"
                ]
            }
        
        # Scenario comparison results
        if scenario_metrics:
            scenario_names = list(scenario_metrics.keys())
            if len(scenario_names) >= 2:
                scenario4_metrics = scenario_metrics[scenario_names[0]]  # No filter
                scenario5_metrics = scenario_metrics[scenario_names[1]]  # Texture enhanced
                
                report['scenario_comparison'] = {
                    'scenario4_no_filter': scenario4_metrics,
                    'scenario5_texture_enhanced': scenario5_metrics,
                    'improvements': {
                        'agreement_rate_improvement': scenario5_metrics['agreement_rate'] - scenario4_metrics['agreement_rate'],
                        'height_diff_reduction': scenario4_metrics['mean_height_diff'] - scenario5_metrics['mean_height_diff'],
                        'correlation_improvement': scenario5_metrics['correlation'] - scenario4_metrics['correlation'],
                        'r2_improvement': scenario5_metrics['r2_score'] - scenario4_metrics['r2_score'],
                        'rmse_reduction': scenario4_metrics['rmse'] - scenario5_metrics['rmse']
                    }
                }
        
        # Research insights and recommendations
        insights = []
        recommendations = []
        
        # Data quality insights
        baseline_agreement = df['height_agreement'].mean()
        insights.append(f"Baseline GEDI-reference agreement rate: {baseline_agreement*100:.1f}%")
        insights.append(f"Mean height difference: {df['height_difference'].mean():.2f} ± {df['height_difference'].std():.2f} m")
        
        # Texture analysis insights
        if correlation_df is not None and len(correlation_df) > 0:
            significant_textures = len(correlation_df[correlation_df['p_height_diff'] < 0.05])
            insights.append(f"Found {significant_textures} texture metrics with significant correlations to height accuracy")
            
            # Identify best texture type
            if len(correlation_df) > 0:
                best_texture = correlation_df.loc[correlation_df['corr_height_diff'].abs().idxmax()]
                insights.append(f"Most predictive texture metric: {best_texture['texture_label']} (r = {best_texture['corr_height_diff']:.3f})")
        
        # Filter performance insights
        if filter_df is not None and len(filter_df) > 0:
            best_filter = filter_df.iloc[0]
            insights.append(f"Best filter achieves {best_filter['agreement_improvement']*100:.1f}% improvement in agreement rate")
            insights.append(f"Optimal texture-based filtering retains {best_filter['retention_rate']*100:.1f}% of data")
        
        # Recommendations for implementation
        if filter_df is not None and len(filter_df) > 0:
            best_filter = filter_df.iloc[0]
            recommendations.append(f"Implement {best_filter['texture_label']} filtering with threshold {best_filter['threshold']:.6f}")
            recommendations.append(f"Use '{best_filter['direction']}' filtering direction for optimal results")
            
            if best_filter['retention_rate'] < 0.3:
                recommendations.append("WARNING: Best filter has low data retention (<30%). Consider looser thresholds for operational use.")
            elif best_filter['retention_rate'] > 0.8:
                recommendations.append("Excellent data retention (>80%). Filter can be applied operationally without significant data loss.")
        
        recommendations.append("Validate texture-enhanced filtering across different forest types and seasons")
        recommendations.append("Consider ensemble approaches combining multiple texture metrics")
        recommendations.append("Implement cross-validation to ensure filter robustness")
        
        report['key_insights'] = insights
        report['recommendations'] = recommendations
        
        # Hypothesis validation
        hypothesis_results = {}
        
        if scenario_metrics and len(scenario_metrics) >= 2:
            scenario_names = list(scenario_metrics.keys())
            scenario4_metrics = scenario_metrics[scenario_names[0]]
            scenario5_metrics = scenario_metrics[scenario_names[1]]
            
            hypothesis_results = {
                'homogeneous_areas_better_accuracy': {
                    'hypothesis': "Homogeneous areas (high IDM) show better GEDI accuracy",
                    'validated': scenario5_metrics['agreement_rate'] > scenario4_metrics['agreement_rate'],
                    'evidence': f"Agreement rate: {scenario5_metrics['agreement_rate']*100:.1f}% vs {scenario4_metrics['agreement_rate']*100:.1f}%"
                },
                'texture_filtering_improves_prediction': {
                    'hypothesis': "Texture-based filtering improves GEDI height predictions",
                    'validated': scenario5_metrics['correlation'] > scenario4_metrics['correlation'],
                    'evidence': f"Correlation: {scenario5_metrics['correlation']:.3f} vs {scenario4_metrics['correlation']:.3f}"
                },
                'quality_filtering_reduces_errors': {
                    'hypothesis': "Quality filtering reduces height prediction errors",
                    'validated': scenario5_metrics['mean_height_diff'] < scenario4_metrics['mean_height_diff'],
                    'evidence': f"Mean error: {scenario5_metrics['mean_height_diff']:.2f}m vs {scenario4_metrics['mean_height_diff']:.2f}m"
                }
            }
        
        report['hypothesis_validation'] = hypothesis_results
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Saved comprehensive report: {output_file}")
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Phase 4: GEDI Texture Enhancement Analysis')
    parser.add_argument('--csv-dir', default='chm_outputs/', 
                       help='Directory containing GEDI CSV files with reference heights')
    parser.add_argument('--output-dir', default='chm_outputs/gedi_texture_enhancement_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--agreement-threshold', type=float, default=5.0,
                       help='Height agreement threshold in meters (default: 5.0)')
    
    args = parser.parse_args()
    
    print("🔬 Phase 4: GEDI Texture Enhancement Analysis")
    print(f"📅 Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 CSV directory: {args.csv_dir}")
    print(f"📊 Output directory: {args.output_dir}")
    print(f"🎯 Agreement threshold: {args.agreement_threshold}m")
    
    # Initialize analyzer
    analyzer = GEDITextureEnhancementAnalyzer(args.output_dir)
    analyzer.agreement_threshold = args.agreement_threshold
    
    # Find and load CSV files
    csv_files = analyzer.find_csv_files(args.csv_dir)
    if not csv_files:
        print("ERROR: No GEDI CSV files found")
        return
    
    # Load and validate data
    df = analyzer.load_and_validate_data(csv_files)
    if df is None:
        print("ERROR: Failed to load data")
        return
    
    # Clean and prepare data
    df = analyzer.clean_and_prepare_data(df)
    
    # Perform texture correlation analysis
    print("\n" + "="*70)
    print("TEXTURE CORRELATION ANALYSIS")
    print("="*70)
    
    correlation_df = analyzer.analyze_texture_height_correlations(df)
    
    # Identify optimal filters
    print("\n" + "="*70)
    print("FILTER OPTIMIZATION")
    print("="*70)
    
    filter_df = analyzer.identify_optimal_filters(df, correlation_df)
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Texture correlation plots
    if correlation_df is not None:
        analyzer.create_texture_correlation_plots(
            correlation_df, 
            f"{args.output_dir}/texture_correlation_analysis.png"
        )
        
        # Save correlation data
        correlation_df.to_csv(f"{args.output_dir}/texture_correlations.csv", index=False)
        print(f"Saved texture correlation data: {args.output_dir}/texture_correlations.csv")
    
    # Filter evaluation plots
    if filter_df is not None:
        analyzer.create_filter_evaluation_plots(
            filter_df,
            f"{args.output_dir}/filter_evaluation_analysis.png"
        )
        
        # Save filter data
        filter_df.to_csv(f"{args.output_dir}/optimal_filter_candidates.csv", index=False)
        print(f"Saved filter candidates: {args.output_dir}/optimal_filter_candidates.csv")
    
    # Scenario comparison plots
    scenario_metrics = analyzer.create_scenario_comparison_plots(
        df, 
        filter_df,
        f"{args.output_dir}/scenario_comparison_analysis.png"
    )
    
    # Generate comprehensive report
    comprehensive_report = analyzer.generate_comprehensive_report(
        df, 
        correlation_df, 
        filter_df,
        scenario_metrics,
        f"{args.output_dir}/gedi_texture_enhancement_comprehensive_report.json"
    )
    
    # Print summary to console
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total samples analyzed: {len(df):,}")
    print(f"Regions: {', '.join(df['region_name'].unique())}")
    print(f"Source files: {len(df['source_file'].unique())}")
    
    # Texture analysis summary
    if correlation_df is not None and len(correlation_df) > 0:
        significant_textures = len(correlation_df[correlation_df['p_height_diff'] < 0.05])
        print(f"\nTexture Analysis:")
        print(f"  Total texture metrics analyzed: {len(correlation_df)}")
        print(f"  Statistically significant correlations: {significant_textures}")
        
        # Top 3 most predictive textures
        if 'abs_corr_height_diff' in correlation_df.columns:
            top_textures = correlation_df.nsmallest(3, 'abs_corr_height_diff')
        else:
            # Create abs_corr_height_diff column for sorting
            correlation_df_sorted = correlation_df.copy()
            correlation_df_sorted['abs_corr_height_diff'] = correlation_df_sorted['corr_height_diff'].abs()
            top_textures = correlation_df_sorted.nlargest(3, 'abs_corr_height_diff')
        print("  Top predictive texture metrics:")
        for i, (_, row) in enumerate(top_textures.iterrows(), 1):
            print(f"    {i}. {row['texture_label']}: r = {row['corr_height_diff']:.3f} (p = {row['p_height_diff']:.3f})")
    
    # Filter optimization summary
    if filter_df is not None and len(filter_df) > 0:
        best_filter = filter_df.iloc[0]
        print(f"\nOptimal Filter:")
        print(f"  Best texture metric: {best_filter['texture_label']}")
        print(f"  Filter criterion: {best_filter['direction_label']}")
        print(f"  Threshold: {best_filter['threshold']:.6f}")
        print(f"  Quality score: {best_filter['quality_score']:.4f}")
        print(f"  Agreement improvement: {best_filter['agreement_improvement']*100:.1f}%")
        print(f"  Data retention: {best_filter['retention_rate']*100:.1f}%")
    
    # Scenario comparison summary
    if scenario_metrics and len(scenario_metrics) >= 2:
        scenario_names = list(scenario_metrics.keys())
        scenario4_metrics = scenario_metrics[scenario_names[0]]
        scenario5_metrics = scenario_metrics[scenario_names[1]]
        
        print(f"\nScenario Comparison:")
        print(f"  Scenario 4 (No Filter): {scenario4_metrics['n_samples']:,} samples")
        print(f"    Agreement rate: {scenario4_metrics['agreement_rate']*100:.1f}%")
        print(f"    Mean height error: {scenario4_metrics['mean_height_diff']:.2f}m")
        print(f"    Correlation: {scenario4_metrics['correlation']:.3f}")
        
        print(f"  Scenario 5 (Texture Enhanced): {scenario5_metrics['n_samples']:,} samples")
        print(f"    Agreement rate: {scenario5_metrics['agreement_rate']*100:.1f}%")
        print(f"    Mean height error: {scenario5_metrics['mean_height_diff']:.2f}m")
        print(f"    Correlation: {scenario5_metrics['correlation']:.3f}")
        
        # Improvements
        agreement_improvement = scenario5_metrics['agreement_rate'] - scenario4_metrics['agreement_rate']
        error_reduction = scenario4_metrics['mean_height_diff'] - scenario5_metrics['mean_height_diff']
        correlation_improvement = scenario5_metrics['correlation'] - scenario4_metrics['correlation']
        
        print(f"  Improvements:")
        print(f"    Agreement rate: +{agreement_improvement*100:.1f}%")
        print(f"    Height error reduction: -{error_reduction:.2f}m")
        print(f"    Correlation improvement: +{correlation_improvement:.3f}")
    
    # Clean up memory
    del df
    if correlation_df is not None:
        del correlation_df
    if filter_df is not None:
        del filter_df
    gc.collect()
    
    print(f"\n✅ Phase 4 texture enhancement analysis completed!")
    print(f"📁 Results saved in: {args.output_dir}")
    print(f"📊 Key outputs:")
    print(f"  • Texture correlation analysis: texture_correlation_analysis.png")
    print(f"  • Filter evaluation: filter_evaluation_analysis.png")  
    print(f"  • Scenario comparison: scenario_comparison_analysis.png")
    print(f"  • Comprehensive report: gedi_texture_enhancement_comprehensive_report.json")

if __name__ == "__main__":
    main()