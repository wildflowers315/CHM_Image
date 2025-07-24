#!/usr/bin/env python3
"""
Phase 3: GEDI Height Correlation Analysis for Pixel-Level CSV Data

This script analyzes correlations between reference heights and various height products
from the GEDI pixel extraction CSV files created in Phase 1-2.

Usage:
    python analysis/gedi_height_correlation_analysis.py --csv-dir chm_outputs/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import argparse
import glob
from pathlib import Path
import json
from datetime import datetime
import sys
import gc

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GEDIHeightCorrelationAnalyzer:
    """Analyzer for GEDI pixel-level height correlation analysis."""
    
    def __init__(self, output_dir="chm_outputs/gedi_height_correlation_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define height columns to analyze
        self.height_columns = [
            'reference_height',  # Ground truth from TIF
            'rh',               # GEDI height quantile
            'ch_potapov2021',   # Global canopy height
            'ch_lang2022',      # Deep learning height
            'ch_tolan2024',     # Advanced height product
            'ch_pauls2024'      # Recent height estimates
        ]
        
        # Human-readable names for plotting
        self.height_labels = {
            'reference_height': 'Reference Height (Ground Truth)',
            'rh': 'GEDI Height (rh98)',
            'ch_potapov2021': 'Potapov et al. 2021',
            'ch_lang2022': 'Lang et al. 2022 (DL)',
            'ch_tolan2024': 'Tolan et al. 2024',
            'ch_pauls2024': 'Pauls et al. 2024'
        }
        
        # Region mapping
        self.regions = {
            'dchm_04hf3': 'Kochi',
            'dchm_05LE4': 'Hyogo', 
            'dchm_09gd4': 'Tochigi'
        }
        
    def find_csv_files(self, csv_dir):
        """Find all GEDI CSV files with reference heights."""
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
        """Load and validate CSV data."""
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
                
                # Check available height columns
                available_heights = [col for col in self.height_columns if col in df.columns]
                print(f"  Available height columns: {available_heights}")
                
                if len(available_heights) >= 2:  # Need at least reference + one other
                    # Display basic statistics
                    for col in available_heights:
                        valid_count = df[col].notna().sum()
                        if valid_count > 0:
                            print(f"    {col}: {valid_count:,} valid values, range: {df[col].min():.1f} to {df[col].max():.1f}")
                    
                    all_data.append(df)
                else:
                    print(f"  WARNING: Insufficient height columns, skipping file")
                    
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
    
    def clean_height_data(self, df):
        """Clean and filter height data."""
        print("\nCleaning height data...")
        
        # Initial count
        initial_count = len(df)
        print(f"  Initial data points: {initial_count:,}")
        
        # Remove rows with missing reference height
        df = df.dropna(subset=['reference_height'])
        print(f"  After removing missing reference heights: {len(df):,}")
        
        # Filter realistic height ranges (>0 and ≤100m for forest canopy, exclude zero reference heights)
        height_mask = (df['reference_height'] > 0) & (df['reference_height'] <= 100)
        df = df[height_mask]
        print(f"  After height range filtering (>0 and ≤100m): {len(df):,}")
        
        # Remove statistical outliers for reference height (3-sigma rule)
        ref_mean = df['reference_height'].mean()
        ref_std = df['reference_height'].std()
        outlier_mask = np.abs(df['reference_height'] - ref_mean) < 3 * ref_std
        df = df[outlier_mask]
        print(f"  After outlier removal (3-sigma): {len(df):,}")
        
        # Clean other height columns
        for col in self.height_columns[1:]:  # Skip reference_height
            if col in df.columns:
                # Count before cleaning
                before_count = df[col].notna().sum()
                
                # Replace unrealistic values with NaN
                df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan
                
                # Count after cleaning
                after_count = df[col].notna().sum()
                if before_count > 0:
                    print(f"    {col}: {before_count:,} -> {after_count:,} valid values")
        
        print(f"  Final cleaned dataset: {len(df):,} points ({len(df)/initial_count*100:.1f}% retention)")
        
        return df
    
    def calculate_correlation_metrics(self, df, col1, col2, region_filter=None):
        """Calculate comprehensive correlation metrics between two columns."""
        # Apply region filter if specified
        if region_filter:
            df_filtered = df[df['region_code'] == region_filter].copy()
        else:
            df_filtered = df.copy()
        
        # Remove NaN values
        valid_mask = df_filtered[col1].notna() & df_filtered[col2].notna()
        
        if valid_mask.sum() < 10:
            return None
        
        x = df_filtered.loc[valid_mask, col1].values
        y = df_filtered.loc[valid_mask, col2].values
        
        try:
            # Basic statistics
            n_samples = len(x)
            
            # Correlation metrics
            pearson_r, pearson_p = stats.pearsonr(x, y)
            spearman_r, spearman_p = stats.spearmanr(x, y)
            
            # Regression metrics (y predicted from x)
            r2 = r2_score(x, y)
            rmse = np.sqrt(mean_squared_error(x, y))
            mae = mean_absolute_error(x, y)
            bias = np.mean(y - x)
            
            # Additional statistics
            slope, intercept, _, _, _ = stats.linregress(x, y)
            
            return {
                'n_samples': n_samples,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'r2_score': r2,
                'rmse': rmse,
                'mae': mae,
                'bias': bias,
                'slope': slope,
                'intercept': intercept,
                'x_mean': np.mean(x),
                'y_mean': np.mean(y),
                'x_std': np.std(x),
                'y_std': np.std(y)
            }
        except Exception as e:
            print(f"Error calculating metrics for {col1} vs {col2}: {e}")
            return None
    
    def create_correlation_matrix(self, df):
        """Create correlation matrix for all height products."""
        print("\nCreating correlation matrix...")
        
        # Get available height columns with sufficient data
        available_heights = []
        for col in self.height_columns:
            if col in df.columns and df[col].notna().sum() > 100:
                available_heights.append(col)
        
        if len(available_heights) < 2:
            print("ERROR: Need at least 2 height columns with sufficient data")
            return None
        
        print(f"Analyzing correlations between {len(available_heights)} height products")
        
        # Calculate correlation matrix
        correlation_data = []
        
        for i, col1 in enumerate(available_heights):
            for j, col2 in enumerate(available_heights):
                if i != j:  # Skip self-correlation
                    metrics = self.calculate_correlation_metrics(df, col1, col2)
                    if metrics:
                        correlation_data.append({
                            'height1': col1,
                            'height2': col2,
                            'height1_label': self.height_labels.get(col1, col1),
                            'height2_label': self.height_labels.get(col2, col2),
                            **metrics
                        })
        
        return pd.DataFrame(correlation_data)
    
    def create_correlation_heatmap(self, correlation_df, output_file):
        """Create correlation heatmap visualization."""
        if correlation_df is None or len(correlation_df) == 0:
            print("No correlation data available for heatmap")
            return
        
        # Get unique height products
        height_products = sorted(list(set(correlation_df['height1'].tolist() + correlation_df['height2'].tolist())))
        
        # Create correlation matrix for heatmap
        corr_matrix = np.zeros((len(height_products), len(height_products)))
        n_samples_matrix = np.zeros((len(height_products), len(height_products)))
        
        # Fill correlation matrix
        for _, row in correlation_df.iterrows():
            i = height_products.index(row['height1'])
            j = height_products.index(row['height2'])
            corr_matrix[i, j] = row['pearson_r']
            n_samples_matrix[i, j] = row['n_samples']
        
        # Fill diagonal with 1.0 (self-correlation)
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Create labels
        labels = [self.height_labels.get(h, h) for h in height_products]
        
        # Create heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Correlation heatmap
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdYlBu_r',
                   center=0,
                   square=True,
                   xticklabels=labels,
                   yticklabels=labels,
                   ax=ax1,
                   cbar_kws={'label': 'Pearson Correlation'})
        ax1.set_title('Height Product Correlation Matrix', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='y', rotation=0)
        
        # Sample size heatmap
        sns.heatmap(n_samples_matrix, 
                   annot=True, 
                   fmt='.0f', 
                   cmap='Greens',
                   square=True,
                   xticklabels=labels,
                   yticklabels=labels,
                   ax=ax2,
                   cbar_kws={'label': 'Sample Size'})
        ax2.set_title('Sample Size Matrix', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved correlation heatmap: {output_file}")
    
    def create_scatter_plots(self, df, output_file):
        """Create scatter plot matrix for height products vs reference."""
        # Get available height columns with reference_height first
        available_heights = ['reference_height'] + [col for col in self.height_columns[1:] 
                                                   if col in df.columns and df[col].notna().sum() > 100]
        
        if len(available_heights) < 2:
            print("Insufficient data for scatter plots")
            return
        
        n_heights = len(available_heights) - 1  # Exclude reference as we compare others to it
        fig, axes = plt.subplots(1, 5, figsize=(25, 6))  # Assume max 6 height products
        axes = axes.flatten()
        
        plot_idx = 0
        for col in available_heights[1:]:  # Skip reference_height
            if plot_idx >= len(axes):
                break
                
            ax = axes[plot_idx]
            
            # Get valid data
            valid_mask = df['reference_height'].notna() & df[col].notna()
            if valid_mask.sum() < 100:
                ax.text(0.5, 0.5, f'Insufficient Data\\nfor {self.height_labels.get(col, col)}', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_xlabel('Reference Height (m)')
                ax.set_ylabel(self.height_labels.get(col, col))
                plot_idx += 1
                continue
            
            x_data = df.loc[valid_mask, 'reference_height']
            y_data = df.loc[valid_mask, col]
            
            # Sample data if too large for visualization
            if len(x_data) > 10000:
                sample_idx = np.random.choice(len(x_data), 10000, replace=False)
                x_data = x_data.iloc[sample_idx]
                y_data = y_data.iloc[sample_idx]
            
            # Create hexbin plot
            hb = ax.hexbin(x_data, y_data, gridsize=25, cmap='viridis', alpha=0.8, mincnt=1)
            
            # Add 1:1 line
            min_val = min(x_data.min(), y_data.min())
            max_val = max(x_data.max(), y_data.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7, label='1:1 Line')
            
            # Calculate and display metrics
            metrics = self.calculate_correlation_metrics(df, 'reference_height', col)
            if metrics:
                # Add regression line
                x_line = np.linspace(min_val, max_val, 100)
                y_line = metrics['slope'] * x_line + metrics['intercept']
                ax.plot(x_line, y_line, 'g-', linewidth=2, alpha=0.8, 
                       label=f'Fit: y = {metrics["slope"]:.3f}x + {metrics["intercept"]:.2f}')
                
                text = (f"R² = {metrics['r2_score']:.3f}\n"
                       f"RMSE = {metrics['rmse']:.2f} m\n"
                       f"Bias = {metrics['bias']:.2f} m\n"
                       f"r = {metrics['pearson_r']:.3f}\n"
                       f"N = {metrics['n_samples']:,}")
                ax.text(0.05, 0.95, text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Reference Height (m)', fontsize=12)
            ax.set_ylabel(self.height_labels.get(col, col), fontsize=12)
            ax.set_title(f'Reference vs {self.height_labels.get(col, col)}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=9)
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Height Products vs Reference Height Correlation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved scatter plots: {output_file}")
    
    def create_regional_comparison(self, df, output_file):
        """Create regional comparison of height correlations."""
        print("\\nCreating regional comparison...")
        
        # Get height products with sufficient data
        height_products = [col for col in self.height_columns[1:] 
                          if col in df.columns and df[col].notna().sum() > 100]
        
        if not height_products:
            print("No height products with sufficient data for regional comparison")
            return None
        
        # Calculate regional correlations
        regional_results = []
        
        for region_code, region_name in self.regions.items():
            region_data = df[df['region_code'] == region_code]
            if len(region_data) < 50:
                continue
            
            print(f"  Processing {region_name}: {len(region_data):,} points")
            
            for height_col in height_products:
                metrics = self.calculate_correlation_metrics(df, 'reference_height', height_col, 
                                                           region_filter=region_code)
                
                if metrics:
                    regional_results.append({
                        'region_code': region_code,
                        'region_name': region_name,
                        'height_product': height_col,
                        'height_label': self.height_labels.get(height_col, height_col),
                        **metrics
                    })
        
        if not regional_results:
            print("No regional results to plot")
            return None
        
        regional_df = pd.DataFrame(regional_results)
        
        # Create regional comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # R² comparison
        r2_pivot = regional_df.pivot(index='height_label', columns='region_name', values='r2_score')
        sns.heatmap(r2_pivot, annot=True, fmt='.3f', cmap='viridis', ax=axes[0,0])
        axes[0,0].set_title('R² Score by Region and Height Product')
        axes[0,0].set_xlabel('Region')
        axes[0,0].set_ylabel('Height Product')
        
        # RMSE comparison
        rmse_pivot = regional_df.pivot(index='height_label', columns='region_name', values='rmse')
        sns.heatmap(rmse_pivot, annot=True, fmt='.2f', cmap='viridis_r', ax=axes[0,1])
        axes[0,1].set_title('RMSE by Region and Height Product (m)')
        axes[0,1].set_xlabel('Region')
        axes[0,1].set_ylabel('Height Product')
        
        # Correlation comparison
        corr_pivot = regional_df.pivot(index='height_label', columns='region_name', values='pearson_r')
        sns.heatmap(corr_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0, ax=axes[1,0])
        axes[1,0].set_title('Pearson Correlation by Region and Height Product')
        axes[1,0].set_xlabel('Region')
        axes[1,0].set_ylabel('Height Product')
        
        # Sample size comparison
        n_pivot = regional_df.pivot(index='height_label', columns='region_name', values='n_samples')
        sns.heatmap(n_pivot, annot=True, fmt='.0f', cmap='Greens', ax=axes[1,1])
        axes[1,1].set_title('Sample Size by Region and Height Product')
        axes[1,1].set_xlabel('Region')
        axes[1,1].set_ylabel('Height Product')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved regional comparison: {output_file}")
        
        return regional_df
    
    def generate_summary_report(self, df, correlation_df, regional_df, output_file):
        """Generate comprehensive summary report."""
        print("\\nGenerating summary report...")
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'dataset_summary': {
                'total_samples': len(df),
                'regions': df['region_name'].value_counts().to_dict(),
                'height_products_available': [col for col in self.height_columns if col in df.columns],
                'source_files': df['source_file'].unique().tolist()
            }
        }
        
        # Overall correlation summary
        if correlation_df is not None and len(correlation_df) > 0:
            ref_corr = correlation_df[correlation_df['height1'] == 'reference_height'].copy()
            
            if len(ref_corr) > 0:
                best_correlations = ref_corr.nlargest(3, 'pearson_r')[['height2', 'pearson_r', 'r2_score', 'rmse']].to_dict('records')
                report['best_height_products'] = best_correlations
                
                # Performance ranking
                ranking = ref_corr.sort_values('pearson_r', ascending=False)[['height2', 'pearson_r', 'r2_score', 'rmse']]
                report['performance_ranking'] = ranking.to_dict('records')
        
        # Regional analysis summary
        if regional_df is not None and len(regional_df) > 0:
            regional_summary = regional_df.groupby('height_product').agg({
                'pearson_r': ['mean', 'std'],
                'r2_score': ['mean', 'std'],
                'rmse': ['mean', 'std'],
                'n_samples': 'sum'
            }).round(3)
            
            # Convert to regular dict for JSON serialization
            regional_summary_dict = {}
            for product in regional_summary.index:
                regional_summary_dict[product] = {
                    'pearson_r_mean': regional_summary.loc[product, ('pearson_r', 'mean')],
                    'pearson_r_std': regional_summary.loc[product, ('pearson_r', 'std')],
                    'r2_mean': regional_summary.loc[product, ('r2_score', 'mean')],
                    'r2_std': regional_summary.loc[product, ('r2_score', 'std')],
                    'rmse_mean': regional_summary.loc[product, ('rmse', 'mean')],
                    'rmse_std': regional_summary.loc[product, ('rmse', 'std')],
                    'total_samples': regional_summary.loc[product, ('n_samples', 'sum')]
                }
            
            report['regional_summary'] = regional_summary_dict
        
        # Statistical insights
        insights = []
        
        # Data quality assessment
        if 'reference_height' in df.columns:
            ref_stats = df['reference_height'].describe()
            insights.append(f"Reference height range: {ref_stats['min']:.1f} to {ref_stats['max']:.1f} m (mean: {ref_stats['mean']:.1f} ± {ref_stats['std']:.1f} m)")
        
        # Best performing product
        if correlation_df is not None and len(correlation_df) > 0:
            ref_corr = correlation_df[correlation_df['height1'] == 'reference_height']
            if len(ref_corr) > 0:
                best_product = ref_corr.loc[ref_corr['pearson_r'].idxmax()]
                insights.append(f"Best performing height product: {best_product['height2']} (r = {best_product['pearson_r']:.3f}, R² = {best_product['r2_score']:.3f})")
        
        # Regional variation
        if regional_df is not None and len(regional_df) > 0:
            region_variation = regional_df.groupby('height_product')['pearson_r'].std().mean()
            insights.append(f"Average cross-region correlation variation: {region_variation:.3f}")
        
        report['key_insights'] = insights
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Saved summary report: {output_file}")
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Phase 3: GEDI Height Correlation Analysis')
    parser.add_argument('--csv-dir', default='chm_outputs/', 
                       help='Directory containing GEDI CSV files with reference heights')
    parser.add_argument('--output-dir', default='chm_outputs/gedi_height_correlation_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    print("🔬 Phase 3: GEDI Height Correlation Analysis")
    print(f"📅 Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 CSV directory: {args.csv_dir}")
    print(f"📊 Output directory: {args.output_dir}")
    
    # Initialize analyzer
    analyzer = GEDIHeightCorrelationAnalyzer(args.output_dir)
    
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
    
    # Clean data
    df = analyzer.clean_height_data(df)
    
    # Perform correlation analysis
    print("\\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    correlation_df = analyzer.create_correlation_matrix(df)
    
    # Create visualizations
    print("\\nCreating visualizations...")
    
    # Correlation heatmap
    analyzer.create_correlation_heatmap(
        correlation_df, 
        f"{args.output_dir}/height_correlation_heatmap.png"
    )
    
    # Scatter plots
    analyzer.create_scatter_plots(
        df,
        f"{args.output_dir}/height_scatter_plots.png"
    )
    
    # Regional comparison
    regional_df = analyzer.create_regional_comparison(
        df,
        f"{args.output_dir}/regional_comparison.png"
    )
    
    # Generate summary report
    summary_report = analyzer.generate_summary_report(
        df, 
        correlation_df, 
        regional_df,
        f"{args.output_dir}/gedi_height_correlation_summary.json"
    )
    
    # Save detailed data
    if correlation_df is not None:
        correlation_df.to_csv(f"{args.output_dir}/detailed_correlations.csv", index=False)
        print(f"Saved detailed correlation data: {args.output_dir}/detailed_correlations.csv")
    
    if regional_df is not None:
        regional_df.to_csv(f"{args.output_dir}/regional_correlations.csv", index=False)
        print(f"Saved regional correlation data: {args.output_dir}/regional_correlations.csv")
    
    # Print summary to console
    print("\\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total samples analyzed: {len(df):,}")
    print(f"Regions: {', '.join(df['region_name'].unique())}")
    print(f"Source files: {len(df['source_file'].unique())}")
    
    if correlation_df is not None and len(correlation_df) > 0:
        ref_corr = correlation_df[correlation_df['height1'] == 'reference_height'].sort_values('pearson_r', ascending=False)
        print("\\nHeight Product Performance Ranking:")
        for i, (_, row) in enumerate(ref_corr.head(5).iterrows(), 1):
            print(f"  {i}. {row['height2']}: r = {row['pearson_r']:.3f}, R² = {row['r2_score']:.3f}, RMSE = {row['rmse']:.2f}m")
    
    # Clean up memory
    del df, correlation_df
    if regional_df is not None:
        del regional_df
    gc.collect()
    
    print(f"\\n✅ Phase 3 analysis completed!")
    print(f"📁 Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()