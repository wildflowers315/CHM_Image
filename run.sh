# single step
# python run_main.py \
# --aoi_path downloads/dchm_09gd4.geojson \
# --year 2022 \
# --start-date 01-01 \
# --end-date 12-31 \
# --steps data_preparation \
# --eval_tif_path downloads/dchm_09gd4.tif

# data preparation
python run_main.py --aoi_path downloads/dchm_09gd4.geojson --year 2022 --start-date 01-01 --end-date 12-31 --steps data_preparation --eval_tif_path downloads/dchm_09gd4.tif

# height analysis
python run_main.py --aoi_path downloads/dchm_09gd4.geojson --year 2022 --start-date 01-01 --end-date 12-31 --steps height_analysis --eval_tif_path downloads/dchm_09gd4.tif

# train predict
python run_main.py --aoi_path downloads/dchm_09gd4.geojson --year 2022 --start-date 01-01 --end-date 12-31 --steps train_predict --eval_tif_path downloads/dchm_09gd4.tif

# evaluate
python run_main.py --aoi_path downloads/dchm_09gd4.geojson --year 2022 --start-date 01-01 --end-date 12-31 --steps evaluate --eval_tif_path downloads/dchm_09gd4.tif

# all steps
# python run_main.py \
# --aoi_path downloads/new_aoi.geojson \
# --year 2022 \
# --start-date 01-01 \
# --end-date 12-31 \
# --steps data_preparation height_analysis train_predict evaluate \
# --eval_tif_path downloads/dchm_09id4.tif