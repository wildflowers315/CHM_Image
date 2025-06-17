#!/bin/bash

# Example commands for running the CHM pipeline with 3D U-Net model

# Single step example
# python run_main.py \
# --aoi_path downloads/dchm_09gd4.geojson \
# --year 2022 \
# --start-date 01-01 \
# --end-date 12-31 \
# --steps data_preparation \
# --eval_tif_path downloads/dchm_09gd4.tif \
# --model 3d_unet \
# --use-patches \
# --patch-size 2560 \
# --patch-overlap 0.1

# Data preparation step
python run_main.py \
--aoi_path downloads/dchm_09gd4.geojson \
--year 2022 \
--start-date 01-01 \
--end-date 12-31 \
--steps data_preparation \
--eval_tif_path downloads/dchm_09gd4.tif \
--model 3d_unet \
--use-patches \
--patch-size 2560 \
--patch-overlap 0.1

# Height analysis step
python run_main.py \
--aoi_path downloads/dchm_09gd4.geojson \
--year 2022 \
--start-date 01-01 \
--end-date 12-31 \
--steps height_analysis \
--eval_tif_path downloads/dchm_09gd4.tif \
--model 3d_unet \
--use-patches \
--patch-size 2560 \
--patch-overlap 0.1

# Train and predict step
python run_main.py \
--aoi_path downloads/dchm_09gd4.geojson \
--year 2022 \
--start-date 01-01 \
--end-date 12-31 \
--steps train_predict \
--eval_tif_path downloads/dchm_09gd4.tif \
--model 3d_unet \
--use-patches \
--patch-size 2560 \
--patch-overlap 0.1

# Evaluation step
python run_main.py \
--aoi_path downloads/dchm_09gd4.geojson \
--year 2022 \
--start-date 01-01 \
--end-date 12-31 \
--steps evaluate \
--eval_tif_path downloads/dchm_09gd4.tif \
--model 3d_unet \
--use-patches \
--patch-size 2560 \
--patch-overlap 0.1

# Run all steps at once (commented out example)
# python run_main.py \
# --aoi_path downloads/dchm_09gd4.geojson \
# --year 2022 \
# --start-date 01-01 \
# --end-date 12-31 \
# --steps data_preparation height_analysis train_predict evaluate \
# --eval_tif_path downloads/dchm_09gd4.tif \
# --model 3d_unet \
# --use-patches \
# --patch-size 2560 \
# --patch-overlap 0.1