# CHM Image Processing

This repository contains code for processing and analyzing Canopy Height Model (CHM) data from various satellite sources, including Google Earth Engine (GEE) integration.

## Features

- Process and align multiple satellite data sources (Sentinel-1, Sentinel-2, ALOS-2)
- Google Earth Engine integration for data preparation
- Generate canopy height predictions using machine learning models (Random Forest, MLP)
- Create comprehensive evaluation reports with visualizations
- Support for large area processing with patch-based analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/wildflowers315/CHM_Image.git
cd CHM_Image

# Install dependencies
pip install -r requirements.txt

# Set up Google Earth Engine authentication
earthengine authenticate
```

## Usage

The workflow consists of four main steps that can be run independently:

### 1. Data Preparation
```bash
python run_main.py data_preparation
```
This step:
- Downloads satellite data from Google Earth Engine
- Processes Sentinel-1, Sentinel-2, and ALOS-2 data
- Creates forest mask using WorldCover data
- Exports training data and image stacks

### 2. Height Analysis
```bash
python run_main.py height_analysis
```
This step:
- Combines height predictions with training data
- Analyzes height distributions
- Generates height statistics

### 3. Training and Prediction
```bash
python run_main.py train_predict
```
This step:
- Trains machine learning model (Random Forest or MLP)
- Generates canopy height predictions
- Applies forest mask to predictions
- Saves prediction results

### 4. Evaluation
```bash
python run_main.py evaluate
```
This step:
- Compares predictions with reference data
- Generates comprehensive PDF evaluation report
- Creates visualizations and statistics

## Configuration

The main parameters can be configured in `run_main.py`:

```python
# Data preparation parameters
year = '2022'
start_date = '01-01'
end_date = '12-31'
gedi_start_date = '2022-06-01'
gedi_end_date = '2022-08-31'
buffer = 10000
clouds_th = 70
ndvi_threshold = 0.35
scale = 10

# Model parameters
model_type = 'RF'  # or 'MLP'
num_trees_rf = 500
min_leaf_pop_rf = 5
bag_frac_rf = 0.5
max_nodes_rf = 1000
```

## Project Structure

```
CHM_Image/
├── .gitignore
├── README.md
├── __init__.py
├── alos2_source.py
├── canopyht_source.py
├── chm_main.py
├── combine_heights.py
├── dem_source.py
├── dl_models.py
├── evaluate_predictions.py
├── evaluation_utils.py
├── for_forest_masking.py
├── for_upload_download.py
├── l2a_gedi_source.py
├── raster_utils.py
├── requirements.txt
├── run_main.py
├── save_evaluation_pdf.py
├── sentinel1_source.py
├── sentinel2_source.py
├── tests
    ├── __init__.py
    ├── test_alos2_source.py
    ├── test_canopyht_source.py
    ├── test_chm_main.py
    ├── test_combine_heights.py
    ├── test_dl_models.py
    ├── test_evaluate_predictions.py
    ├── test_forest_masking.py
    ├── test_gedi_source.py
    ├── test_new_random_sampling.py
    ├── test_save_evaluation_pdf.py
    ├── test_save_evaluation_pdf_new.py
    ├── test_sentinel1_source.py
    ├── test_sentinel2_source.py
    └── test_train_predict_map.py
├── train_predict_map.py
└── utils.pyncies
```

## Dependencies

- Python 3.10+
- Google Earth Engine Python API
- PyTorch
- Rasterio
- NumPy
- Pandas
- Matplotlib
- ReportLab
- scikit-learn

## Data Requirements

- Area of Interest (AOI) in GeoJSON format
- GEDI data for training
- Sentinel-1, Sentinel-2, and ALOS-2 data (accessed via GEE)
- WorldCover data for forest masking
- Reference CHM data for evaluation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Earth Engine for data access
- Sentinel-1, Sentinel-2, and ALOS-2 data providers
- GEDI data for validation
- WorldCover for forest masking
- Contributors and maintainers of the open-source libraries used in this project


### Converting GeoTIFF to GeoJSON

The repository includes a utility function to convert GeoTIFF files to GeoJSON format. This is particularly useful for visualizing and analyzing raster data in GIS applications.

To convert a GeoTIFF file to GeoJSON:

```python
from utils import geotiff_to_geojson

# If your GeoTIFF is in the Downloads folder
geotiff_path = "downloads\dchm_09gd4.tif"
geojson_path = geotiff_to_geojson(geotiff_path)
print(f"Created GeoJSON file at: {geojson_path}")
```

