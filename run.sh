#!/bin/bash

# =============================================================================
# CHM (Canopy Height Mapping) Pipeline - Complete Workflow Examples
# =============================================================================
# This script demonstrates both traditional workflow (run_main.py) and 
# new unified training system (train_predict_map.py)

# =============================================================================
# TRADITIONAL WORKFLOW - Using run_main.py (4-step process)
# =============================================================================

echo "=== Traditional CHM Workflow Examples ==="

# --- FULL WORKFLOW: Non-temporal mode (Paul's 2024 methodology) ---
# Step 1-4: Complete pipeline from data collection to evaluation
python run_main.py \
    --aoi_path downloads/dchm_09gd4.geojson \
    --year 2022 \
    --start-date 01-01 \
    --end-date 12-31 \
    --eval_tif_path downloads/dchm_09gd4.tif \
    --use-patches \
    --patch-size 2560 \
    --patch-overlap 0.0 \
    --model 3d_unet \
    --steps data_preparation height_analysis train_predict evaluate

# --- FULL WORKFLOW: Temporal mode (Paul's 2025 methodology) ---
# Step 1-4: Complete pipeline with 12-monthly temporal compositing
# python run_main.py \
#     --aoi_path downloads/dchm_09gd4.geojson \
#     --year 2022 \
#     --start-date 01-01 \
#     --end-date 12-31 \
#     --eval_tif_path downloads/dchm_09gd4.tif \
#     --use-patches \
#     --patch-size 2560 \
#     --patch-overlap 0.0 \
#     --model 3d_unet \
#     --temporal-mode \
#     --monthly-composite median \
#     --steps data_preparation height_analysis train_predict evaluate

# --- STEP-BY-STEP WORKFLOW ---
# Step 1: Data preparation (GEE export) - Non-temporal
# python run_main.py \
#     --aoi_path downloads/dchm_09gd4.geojson \
#     --year 2022 \
#     --start-date 01-01 \
#     --end-date 12-31 \
#     --eval_tif_path downloads/dchm_09gd4.tif \
#     --use-patches \
#     --patch-size 2560 \
#     --patch-overlap 0.0 \
#     --model 3d_unet \
#     --steps data_preparation

# Step 1: Data preparation (GEE export) - Temporal
# python run_main.py \
#     --aoi_path downloads/dchm_09gd4.geojson \
#     --year 2022 \
#     --start-date 01-01 \
#     --end-date 12-31 \
#     --eval_tif_path downloads/dchm_09gd4.tif \
#     --use-patches \
#     --patch-size 2560 \
#     --patch-overlap 0.0 \
#     --model 3d_unet \
#     --temporal-mode \
#     --monthly-composite median \
#     --steps data_preparation

# Step 2: Height analysis
# python run_main.py \
#     --aoi_path downloads/dchm_09gd4.geojson \
#     --year 2022 \
#     --start-date 01-01 \
#     --end-date 12-31 \
#     --eval_tif_path downloads/dchm_09gd4.tif \
#     --use-patches \
#     --patch-size 2560 \
#     --patch-overlap 0.0 \
#     --model 3d_unet \
#     --steps height_analysis

# Step 3: Train and predict
# python run_main.py \
#     --aoi_path downloads/dchm_09gd4.geojson \
#     --year 2022 \
#     --start-date 01-01 \
#     --end-date 12-31 \
#     --eval_tif_path downloads/dchm_09gd4.tif \
#     --use-patches \
#     --patch-size 2560 \
#     --patch-overlap 0.0 \
#     --model 3d_unet \
#     --steps train_predict

# Step 4: Evaluation with PDF report
# python run_main.py \
#     --aoi_path downloads/dchm_09gd4.geojson \
#     --year 2022 \
#     --start-date 01-01 \
#     --end-date 12-31 \
#     --eval_tif_path downloads/dchm_09gd4.tif \
#     --use-patches \
#     --patch-size 2560 \
#     --patch-overlap 0.0 \
#     --model 3d_unet \
#     --steps evaluate

# =============================================================================
# UNIFIED TRAINING SYSTEM - Using train_predict_map.py (Direct training)
# =============================================================================

echo "=== Unified Training System Examples ==="

# --- NON-TEMPORAL MODEL TRAINING ---
# Random Forest with non-temporal data (31 bands)
# python train_predict_map.py \
#     --patch-path "chm_outputs/dchm_09gd4_bandNum31_scale10_patch0000.tif" \
#     --model rf \
#     --output-dir chm_outputs/unified_rf \
#     --epochs 10

# MLP with non-temporal data (31 bands)
# python train_predict_map.py \
#     --patch-path "chm_outputs/dchm_09gd4_bandNum31_scale10_patch0000.tif" \
#     --model mlp \
#     --output-dir chm_outputs/unified_mlp \
#     --epochs 50 \
#     --learning-rate 1e-3 \
#     --batch-size 64

# 2D U-Net with non-temporal data (31 bands)
# python train_predict_map.py \
#     --patch-path "chm_outputs/dchm_09gd4_bandNum31_scale10_patch0000.tif" \
#     --model 2d_unet \
#     --output-dir chm_outputs/unified_2d_unet \
#     --epochs 30 \
#     --learning-rate 1e-3 \
#     --base-channels 32 \
#     --generate-prediction

# --- TEMPORAL MODEL TRAINING ---
# Random Forest with temporal data (194 bands)
# python train_predict_map.py \
#     --patch-path "chm_outputs/dchm_09gd4_temporal_bandNum196_scale10_patch0000.tif" \
#     --model rf \
#     --output-dir chm_outputs/temporal_rf \
#     --epochs 10

# MLP with temporal data (194 bands) - Best performing traditional model
# python train_predict_map.py \
#     --patch-path "chm_outputs/dchm_09gd4_temporal_bandNum196_scale10_patch0000.tif" \
#     --model mlp \
#     --output-dir chm_outputs/temporal_mlp \
#     --epochs 100 \
#     --learning-rate 1e-3 \
#     --batch-size 32

# 3D U-Net with temporal data (194 bands) - Paul's 2025 methodology
# python train_predict_map.py \
#     --patch-path "chm_outputs/dchm_09gd4_temporal_bandNum196_scale10_patch0000.tif" \
#     --model 3d_unet \
#     --output-dir chm_outputs/temporal_3d_unet \
#     --epochs 20 \
#     --learning-rate 1e-4 \
#     --base-channels 16 \
#     --generate-prediction

# --- MODEL COMPARISON STUDY ---
# Train all models for comparison (requires both patch types)
echo "=== Model Comparison Study ==="

# # Non-temporal models
# python train_predict_map.py --patch-path "chm_outputs/dchm_09gd4_bandNum31_scale10_patch0000.tif" --model rf --output-dir chm_outputs/comparison/rf_non_temporal
# python train_predict_map.py --patch-path "chm_outputs/dchm_09gd4_bandNum31_scale10_patch0000.tif" --model mlp --output-dir chm_outputs/comparison/mlp_non_temporal
# python train_predict_map.py --patch-path "chm_outputs/dchm_09gd4_bandNum31_scale10_patch0000.tif" --model 2d_unet --output-dir chm_outputs/comparison/2d_unet --generate-prediction

# # Temporal models
# python train_predict_map.py --patch-path "chm_outputs/dchm_09gd4_temporal_bandNum196_scale10_patch0000.tif" --model rf --output-dir chm_outputs/comparison/rf_temporal
# python train_predict_map.py --patch-path "chm_outputs/dchm_09gd4_temporal_bandNum196_scale10_patch0000.tif" --model mlp --output-dir chm_outputs/comparison/mlp_temporal
# python train_predict_map.py --patch-path "chm_outputs/dchm_09gd4_temporal_bandNum196_scale10_patch0000.tif" --model 3d_unet --output-dir chm_outputs/comparison/3d_unet --generate-prediction

# =============================================================================
# ADVANCED CONFIGURATION EXAMPLES
# =============================================================================

# --- High-resolution training (scale 10m) ---
# python run_main.py \
#     --aoi_path downloads/dchm_09gd4.geojson \
#     --year 2022 \
#     --start-date 01-01 \
#     --end-date 12-31 \
#     --eval_tif_path downloads/dchm_09gd4.tif \
#     --use-patches \
#     --patch-size 2560 \
#     --patch-overlap 0.1 \
#     --model 3d_unet \
#     --temporal-mode \
#     --monthly-composite median \
#     --scale 10 \
#     --steps data_preparation

# --- Custom U-Net parameters ---
# python train_predict_map.py \
#     --patch-path "chm_outputs/temporal_patch.tif" \
#     --model 3d_unet \
#     --output-dir chm_outputs/custom_3d_unet \
#     --epochs 50 \
#     --learning-rate 5e-4 \
#     --weight-decay 1e-4 \
#     --base-channels 64 \
#     --huber-delta 1.5 \
#     --shift-radius 2 \
#     --generate-prediction

# =============================================================================
# DATA PREPARATION FOR UNIFIED TRAINING
# =============================================================================

echo "=== Data Preparation for Unified Training ==="

# 1. Generate non-temporal patches (for RF, MLP, 2D U-Net)
# python run_main.py \
#     --aoi_path downloads/dchm_09gd4.geojson \
#     --year 2022 \
#     --start-date 01-01 \
#     --end-date 12-31 \
#     --eval_tif_path downloads/dchm_09gd4.tif \
#     --use-patches \
#     --patch-size 2560 \
#     --patch-overlap 0.0 \
#     --model 3d_unet \
#     --steps data_preparation

# 2. Generate temporal patches (for RF, MLP, 3D U-Net with temporal data)
# python run_main.py \
#     --aoi_path downloads/dchm_09gd4.geojson \
#     --year 2022 \
#     --start-date 01-01 \
#     --end-date 12-31 \
#     --eval_tif_path downloads/dchm_09gd4.tif \
#     --use-patches \
#     --patch-size 2560 \
#     --patch-overlap 0.0 \
#     --model 3d_unet \
#     --temporal-mode \
#     --monthly-composite median \
#     --steps data_preparation

# =============================================================================
# USAGE NOTES
# =============================================================================

# Expected Data Structure:
# - Non-temporal patches: ~31 bands (S1: 2, S2: 11, ALOS2: 2, DEM: ~5, CH: ~4, etc.)
# - Temporal patches: ~196 bands (S1: 24, S2: 132, ALOS2: 24, DEM: ~5, CH: ~4, etc.)
# 
# Model Compatibility:
# - RF/MLP: Works with both temporal and non-temporal data
# - 2D U-Net: Non-temporal data only
# - 3D U-Net: Temporal data only (with intelligent fallback)
# 
# Performance Ranking (based on test results):
# 1. MLP (temporal): R² = 0.391, RMSE = 5.95m ⭐⭐
# 2. RF (non-temporal): R² = 0.175, RMSE = 6.92m ⭐
# 3. U-Net models: Need more training/tuning
#
# Environment Setup:
# source chm_env/bin/activate  # Activate Python environment
# earthengine authenticate     # Required for GEE data collection

echo "=== End of run.sh ==="