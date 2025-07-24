#!/usr/bin/env python3
"""
Phase 5: GEDI Quality Classifier Training

This script trains a machine learning model (RandomForestClassifier) to predict the
quality of GEDI footprints based on texture features, satellite data, and auxiliary
height products. The goal is to create a robust filter for unreliable GEDI data
points prior to their use in canopy height model training.

Usage:
    python analysis/train_gedi_quality_classifier.py --csv-dir chm_outputs/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import os
import argparse
import glob
from pathlib import Path
import json
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GEDIQualityClassifierTrainer:
    """Trains a classifier to predict GEDI data quality."""

    def __init__(self, output_dir="chm_outputs/gedi_quality_classifier"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.agreement_threshold = 10.0  # meters

        # Define texture metrics available in the dataset
        self.texture_metrics = [
            'mean_asm', 'mean_contrast', 'mean_corr', 'mean_var', 'mean_idm',
            'mean_savg', 'mean_ent', 'median_asm', 'median_contrast',
            'median_corr', 'median_var', 'median_idm', 'median_savg', 'median_ent'
        ]
        # Google Embedding Bands
        self.embedding_bands = [f'A{i:02d}' for i in range(64)]

    def find_csv_files(self, csv_dir):
        """Find all GEDI CSV files with reference heights and texture data."""
        print(f"Searching for CSV files in: {csv_dir}")
        patterns = ["*gedi_embedding*with_reference.csv", "*gedi*reference*.csv"]
        csv_files = []
        for pattern in patterns:
            files = glob.glob(os.path.join(csv_dir, pattern))
            csv_files.extend(files)
        csv_files = sorted(list(set(csv_files)))
        print(f"Found {len(csv_files)} CSV files.")
        return csv_files

    def load_and_prepare_data(self, csv_files):
        """Load, clean, and prepare data for classification."""
        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                all_data.append(df)
            except Exception as e:
                print(f"  ERROR loading {csv_file}: {e}")

        if not all_data:
            print("ERROR: No valid data loaded")
            return None

        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Combined dataset: {len(combined_df):,} rows")

        # --- Data Cleaning and Feature Engineering ---
        print("Cleaning data and creating target variable...")
        # Remove rows with missing essential data
        df = combined_df.dropna(subset=['reference_height', 'rh'])
        # Filter for realistic height ranges
        height_mask = ((df['reference_height'] > 0) & (df['reference_height'] <= 100) &
                      (df['rh'] > 0) & (df['rh'] <= 100))
        df = df[height_mask].copy()

        # Create the binary target variable
        df['height_difference'] = np.abs(df['reference_height'] - df['rh'])
        df['within_agreement'] = (df['height_difference'] <= self.agreement_threshold).astype(int)

        print(f"Target variable 'within_agreement' created.")
        agreement_rate = df['within_agreement'].mean()
        print(f"  - Agreement rate (target=1): {agreement_rate:.2%}")
        print(f"  - Disagreement rate (target=0): {1 - agreement_rate:.2%}")
        
        # Handle NaNs in feature columns
        feature_cols = self.texture_metrics + self.embedding_bands
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        return df

    def train_classifier(self, df):
        """Train the RandomForestClassifier."""
        print("\nTraining GEDI Quality Classifier...")

        # Define features (X) and target (y)
        # feature_cols = [col for col in self.texture_metrics + self.embedding_bands if col in df.columns]
        feature_cols = [col for col in self.texture_metrics if col in df.columns]

        X = df[feature_cols]
        y = df['within_agreement']

        print(f"Training with {len(feature_cols)} features on {len(df)} samples.")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Initialize and train the classifier
        # Using class_weight='balanced' to handle imbalanced target variable
        classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        classifier.fit(X_train, y_train)

        print("Classifier training complete.")
        return classifier, X_test, y_test

    def evaluate_classifier(self, classifier, X_test, y_test):
        """Evaluate the classifier and generate reports."""
        print("\nEvaluating classifier performance...")
        y_pred = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)[:, 1]

        # --- Performance Metrics ---
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, target_names=['Disagree (0)', 'Agree (1)'])

        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print("Classification Report:")
        print(report)

        # --- Feature Importance ---
        feature_importances = pd.DataFrame({
            'feature': X_test.columns,
            'importance': classifier.feature_importances_
        }).sort_values('importance', ascending=False)

        # --- Visualizations ---
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Disagree', 'Agree'], yticklabels=['Disagree', 'Agree'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f"{self.output_dir}/confusion_matrix.png")
        plt.close()
        print(f"Saved confusion matrix plot to {self.output_dir}/")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/roc_curve.png")
        plt.close()
        print(f"Saved ROC curve plot to {self.output_dir}/")

        # Feature Importance Plot
        plt.figure(figsize=(12, 10))
        sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/feature_importance.png")
        plt.close()
        print(f"Saved feature importance plot to {self.output_dir}/")

        return {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "feature_importances": feature_importances.to_dict('records')
        }

    def save_model(self, model, filename="gedi_quality_classifier.joblib"):
        """Save the trained model to disk."""
        path = os.path.join(self.output_dir, filename)
        joblib.dump(model, path)
        print(f"\nTrained model saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description='Phase 5: GEDI Quality Classifier Training')
    parser.add_argument('--csv-dir', default='chm_outputs/',
                       help='Directory containing GEDI CSV files with reference heights')
    parser.add_argument('--output-dir', default='chm_outputs/gedi_quality_classifier',
                       help='Output directory for the trained model and analysis results')
    args = parser.parse_args()

    print("🔬 Phase 5: GEDI Quality Classifier Training")
    print(f"📅 Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    trainer = GEDIQualityClassifierTrainer(args.output_dir)

    # Load and prepare data
    csv_files = trainer.find_csv_files(args.csv_dir)
    if not csv_files:
        print("ERROR: No GEDI CSV files found.")
        return

    df = trainer.load_and_prepare_data(csv_files)
    if df is None:
        print("ERROR: Failed to load or prepare data.")
        return

    # Train the classifier
    classifier, X_test, y_test = trainer.train_classifier(df)

    # Evaluate the classifier
    trainer.evaluate_classifier(classifier, X_test, y_test)

    # Save the final model
    trainer.save_model(classifier)

    print(f"\n✅ Phase 5 training and evaluation completed!")
    print(f"📁 Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()