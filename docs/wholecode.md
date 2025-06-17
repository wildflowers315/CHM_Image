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
└── utils.py


/.gitignore:
--------------------------------------------------------------------------------
 1 | # Compiled class file
 2 | *.class
 3 | 
 4 | # Log file
 5 | *.log
 6 | 
 7 | # BlueJ files
 8 | *.ctxt
 9 | 
10 | # Mobile Tools for Java (J2ME)
11 | .mtj.tmp/
12 | 
13 | # Package Files #
14 | *.jar
15 | *.war
16 | *.nar
17 | *.ear
18 | *.zip
19 | *.tar.gz
20 | *.rar
21 | 
22 | # virtual machine crash logs, see http://www.java.com/en/download/help/error_hotspot.xml
23 | hs_err_pid*
24 | 
25 | # Pycache
26 | *__pycache__/
27 | *.pytest_cache*
28 | 
29 | # downloaded files
30 | downloads/*
31 | outputs/*
32 | chm_outputs/*
33 | chm_outputs/
34 | chm_outputs*
35 | chm_outputs/**/*
36 | 
37 | */evaluation/*
38 | */old*
39 | */old/*
40 | # docs/*
41 | 
42 | .cursorrules
43 | 


--------------------------------------------------------------------------------
/README.md:
--------------------------------------------------------------------------------
1 | # CHM_Image
2 | Canopy Height Mapper with Image-based model. 
3 | 


--------------------------------------------------------------------------------
/__init__.py:
--------------------------------------------------------------------------------
https://raw.githubusercontent.com/wildflowers315/CHM_Image/4635724dbf5dd49b590f69f0fac781123bfd571d/__init__.py


--------------------------------------------------------------------------------
/alos2_source.py:
--------------------------------------------------------------------------------
  1 | """
  2 | Module for retrieving and processing ALOS-2 PALSAR SAR data from Google Earth Engine.
  3 | """
  4 | 
  5 | import ee
  6 | from typing import Union, List
  7 | 
  8 | def get_alos2_data(
  9 |     aoi: ee.Geometry,
 10 |     year: int,
 11 |     start_date: str = "01-01",
 12 |     end_date: str = "12-31",
 13 |     include_texture: bool = True,
 14 |     speckle_filter: bool = True,
 15 | ) -> ee.Image:
 16 |     """
 17 |     Get ALOS-2 PALSAR data for the specified area and time period.
 18 |     
 19 |     Args:
 20 |         aoi: Area of interest as Earth Engine Geometry
 21 |         year: Year for analysis
 22 |         start_date: Start date for ALOS-2 data (format: MM-DD, default: 01-01)
 23 |         end_date: End date for ALOS-2 data (format: MM-DD, default: 12-31)
 24 |         include_texture: Whether to include texture metrics (GLCM)
 25 |         speckle_filter: Whether to apply speckle filtering
 26 |     
 27 |     Returns:
 28 |         ee.Image: Processed ALOS-2 PALSAR data
 29 |     """
 30 |     # Format dates properly for Earth Engine
 31 |     start_date_ee = ee.Date(f'{year}-{start_date}')
 32 |     end_date_ee = ee.Date(f'{year}-{end_date}')
 33 |     
 34 |     # Import ALOS-2 PALSAR dataset
 35 |     # Using the ALOS/PALSAR/YEARLY collection (annual mosaics)
 36 |     # alos = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR')"JAXA/ALOS/PALSAR/YEARLY/SAR"
 37 |     alos = ee.ImageCollection("JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR")
 38 |     alos.select(['HH', 'HV'])  # Select HH and HV bands
 39 |     
 40 |     # Filter by date and region
 41 |     alos_filtered = alos.filterDate(start_date_ee, end_date_ee) \
 42 |                        .filterBounds(aoi)
 43 |     
 44 |     # Apply pre-processing
 45 |     def preprocess_sar(img):
 46 |         # Extract date information
 47 |         date = ee.Date(img.get('system:time_start'))
 48 |         
 49 |         # Basic preprocessing - convert to natural units (from dB)
 50 |         # HH and HV bands are stored in dB, convert to natural values for processing
 51 |         hh = ee.Image(10.0).pow(img.select(['HH']).divide(10.0))
 52 |         hv = ee.Image(10.0).pow(img.select(['HV']).divide(10.0))
 53 |                 
 54 |         # Ratio (HH/HV)
 55 |         ratio = hh.divide(hv).rename('ALOS2_ratio')
 56 |         
 57 |         # Normalized difference (HH-HV)/(HH+HV)
 58 |         ndri = hh.subtract(hv).divide(hh.add(hv)).rename('ALOS2_ndri')
 59 |         
 60 |         # Combine all bands
 61 |         processed = img
 62 |         
 63 |         processed = processed.addBands([hh, hv, ratio, ndri])
 64 |         
 65 |         return processed
 66 |     
 67 |     # Apply preprocessing to all images
 68 |     # alos_processed = alos_filtered.map(preprocess_sar)
 69 |     
 70 |     # Apply speckle filtering if requested
 71 |     if speckle_filter:
 72 |         def apply_speckle_filter(img):
 73 |             # Use Refined Lee filter
 74 |             kernel_size = 3  # Use 3x3 kernel
 75 |             hh_filtered = img.select('HH').focal_mean(kernel_size, 'square')
 76 |             hv_filtered = img.select('HV').focal_mean(kernel_size, 'square')
 77 |             
 78 |             # Replace original bands with filtered ones
 79 |             return img.addBands([
 80 |                 hh_filtered.rename('HH_filtered'),
 81 |                 hv_filtered.rename('HV_filtered')
 82 |             ], None, True)
 83 |         
 84 |         alos_processed = alos_processed.map(apply_speckle_filter)
 85 |     
 86 |     # Add texture metrics if requested
 87 |     if include_texture:
 88 |         def add_texture(img):
 89 |             # Define windows sizes for texture calculation
 90 |             windows = [3, 5]
 91 |             
 92 |             texture_bands = []
 93 |             
 94 |             for window in windows:
 95 |                 # Calculate GLCM texture metrics for HH and HV bands
 96 |                 glcm_hh = img.select('HH').glcmTexture(window)
 97 |                 glcm_hv = img.select('HV').glcmTexture(window)
 98 |                 
 99 |                 # Rename bands to include window size
100 |                 glcm_hh = glcm_hh.rename([
101 |                     f'HH_contrast_{window}', f'HH_dissimilarity_{window}', 
102 |                     f'HH_homogeneity_{window}', f'HH_ASM_{window}', 
103 |                     f'HH_energy_{window}', f'HH_max_{window}', 
104 |                     f'HH_entropy_{window}', f'HH_correlation_{window}'
105 |                 ])
106 |                 
107 |                 glcm_hv = glcm_hv.rename([
108 |                     f'HV_contrast_{window}', f'HV_dissimilarity_{window}', 
109 |                     f'HV_homogeneity_{window}', f'HV_ASM_{window}', 
110 |                     f'HV_energy_{window}', f'HV_max_{window}', 
111 |                     f'HV_entropy_{window}', f'HV_correlation_{window}'
112 |                 ])
113 |                 
114 |                 texture_bands.extend([glcm_hh, glcm_hv])
115 |             
116 |             # Add texture bands to the image
117 |             return img.addBands(texture_bands)
118 |         
119 |         alos_processed = alos_processed.map(add_texture)
120 |     
121 |     alos_processed = alos_filtered.select(['HH', 'HV'],['ALOS2_HH', 'ALOS2_HV'])
122 |     # Create median composite
123 |     alos_median = alos_processed.median()
124 |     
125 |     # Clip to area of interest
126 |     alos_median = alos_median.clip(aoi)
127 |     
128 |     # Convert HH and HV bands back to dB for final output
129 |     # This makes it consistent with typical SAR data representation
130 |     def convert_to_db(img):
131 |         hh_db = ee.Image(10).multiply(img.select('HH').log10()).rename('HH_dB')
132 |         hv_db = ee.Image(10).multiply(img.select('HV').log10()).rename('HV_dB')
133 |         return img.addBands([hh_db, hv_db])
134 |     
135 |     # alos_median = convert_to_db(alos_median)
136 |     
137 |     return alos_median
138 | 
139 | def get_alos2_mosaic(
140 |     aoi: ee.Geometry,
141 |     years: List[int] = None,
142 |     include_texture: bool = True,
143 |     speckle_filter: bool = True,
144 | ) -> ee.Image:
145 |     """
146 |     Get multi-year ALOS-2 PALSAR mosaic for the specified area.
147 |     
148 |     Args:
149 |         aoi: Area of interest as Earth Engine Geometry
150 |         years: List of years to include (default: most recent 3 years)
151 |         include_texture: Whether to include texture metrics
152 |         speckle_filter: Whether to apply speckle filtering
153 |     
154 |     Returns:
155 |         ee.Image: Multi-year ALOS-2 PALSAR mosaic
156 |     """
157 |     # If years not specified, use most recent 3 years
158 |     if years is None:
159 |         # Get current date using JavaScript Date
160 |         # ee.Date() without arguments gives the current date/time
161 |         current_year = ee.Date().get('year').subtract(1)  # Last complete year
162 |         years = [
163 |             current_year.subtract(2).getInfo(), 
164 |             current_year.subtract(1).getInfo(), 
165 |             current_year.getInfo()
166 |         ]
167 |     
168 |     # Get data for each year
169 |     year_images = []
170 |     for year in years:
171 |         year_img = get_alos2_data(
172 |             aoi=aoi,
173 |             year=year,
174 |             include_texture=include_texture,
175 |             speckle_filter=speckle_filter
176 |         )
177 |         year_images.append(year_img)
178 |     
179 |     # Create a collection and reduce to median
180 |     alos_collection = ee.ImageCollection.fromImages(year_images)
181 |     alos_mosaic = alos_collection.median()
182 |     
183 |     return alos_mosaic.clip(aoi)
184 | 
185 | # Example usage:
186 | # ee.Initialize()
187 | # aoi = ee.Geometry.Rectangle([139.5, 35.6, 140.0, 36.0])
188 | # alos_data = get_alos2_data(aoi, 2020)
189 | # multi_year_mosaic = get_alos2_mosaic(aoi, [2018, 2019, 2020])


--------------------------------------------------------------------------------
/canopyht_source.py:
--------------------------------------------------------------------------------
 1 | import ee
 2 | 
 3 | def get_canopyht_data(
 4 |     aoi: ee.Geometry,
 5 |     # year: int,
 6 |     # start_date: str,
 7 |     # end_date: str
 8 | ) -> ee.Image:
 9 |     
10 |     # https://github.com/AI4Forest/Global-Canopy-Height-Map/tree/main
11 |     ch_pauls2024 = ee.ImageCollection('projects/worldwidemap/assets/canopyheight2020') \
12 |         .filterBounds(aoi).mosaic().select([0],['ch_pauls2024']).divide(100)  # Convert cm to m
13 |     
14 |     # https://gee-community-catalog.org/projects/meta_trees/#dataset-citation
15 |     ch_tolan2024 = ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight") \
16 |         .filterBounds(aoi).mosaic().select([0],['ch_tolan2024'])
17 |     # treenotree = canopy_ht_2023.updateMask(canopy_ht_2023.gte(0))
18 |     
19 |     # https://gee-community-catalog.org/projects/canopy/#earth-engine-snippet
20 |     ch_lang2022 = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1") \
21 |         .select([0],['ch_lang2022'])
22 |     standard_deviation_2022 = ee.Image("users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1") \
23 |         .select([0],['ch_lang2022_stddev'])
24 |     
25 |     # https://www.sciencedirect.com/science/article/pii/S0034425720305381
26 |     ch_potapov2021 = ee.ImageCollection("users/potapovpeter/GEDI_V27") \
27 |         .filterBounds(aoi).mosaic().select([0],['ch_potapov2021']) # 30m resolution
28 |     
29 |     # Merge the images
30 |     canopy_ht = ch_pauls2024.addBands(ch_tolan2024) \
31 |         .addBands(ch_lang2022) \
32 |         .addBands(ch_potapov2021) 
33 |         # .addBands(standard_deviation_2022) \
34 | 
35 |     return canopy_ht.clip(aoi)


--------------------------------------------------------------------------------
/chm_main.py:
--------------------------------------------------------------------------------
  1 | import ee
  2 | import geemap
  3 | import numpy as np
  4 | import json
  5 | import os
  6 | from pathlib import Path
  7 | import pandas as pd
  8 | import datetime
  9 | import argparse
 10 | from typing import Union, List, Dict, Any
 11 | import time
 12 | from tqdm import tqdm
 13 | import rasterio
 14 | from rasterio.transform import from_origin
 15 | from rasterio.crs import CRS
 16 | 
 17 | # Import custom functions
 18 | from l2a_gedi_source import get_gedi_data
 19 | from sentinel1_source import get_sentinel1_data
 20 | from sentinel2_source import get_sentinel2_data
 21 | from for_forest_masking import apply_forest_mask, create_forest_mask
 22 | from alos2_source import get_alos2_data
 23 | from new_random_sampling import create_training_data, generate_sampling_sites
 24 | from canopyht_source import get_canopyht_data
 25 | from dem_source import get_dem_data
 26 | 
 27 | def load_aoi(aoi_path: str) -> ee.Geometry:
 28 |     """
 29 |     Load AOI from GeoJSON file. Handles both simple Polygon/MultiPolygon and FeatureCollection formats.
 30 |     
 31 |     Args:
 32 |         aoi_path: Path to GeoJSON file
 33 |     
 34 |     Returns:
 35 |         ee.Geometry: Earth Engine geometry object
 36 |     """
 37 |     if not os.path.exists(aoi_path):
 38 |         raise FileNotFoundError(f"AOI file not found: {aoi_path}")
 39 |     
 40 |     with open(aoi_path, 'r') as f:
 41 |         geojson_data = json.load(f)
 42 |     
 43 |     def create_geometry(geom_type: str, coords: List) -> ee.Geometry:
 44 |         """Helper function to create ee.Geometry objects."""
 45 |         if geom_type == 'Polygon':
 46 |             return ee.Geometry.Polygon(coords)
 47 |         elif geom_type == 'MultiPolygon':
 48 |             # MultiPolygon coordinates are nested one level deeper than Polygon
 49 |             return ee.Geometry.MultiPolygon(coords[0])
 50 |         else:
 51 |             raise ValueError(f"Unsupported geometry type: {geom_type}")
 52 |     
 53 |     # Handle FeatureCollection
 54 |     if geojson_data['type'] == 'FeatureCollection':
 55 |         if not geojson_data['features']:
 56 |             raise ValueError("Empty FeatureCollection")
 57 |         
 58 |         # Get the first feature's geometry
 59 |         geometry = geojson_data['features'][0]['geometry']
 60 |         return create_geometry(geometry['type'], geometry['coordinates'])
 61 |     
 62 |     # Handle direct Polygon/MultiPolygon
 63 |     elif geojson_data['type'] in ['Polygon', 'MultiPolygon']:
 64 |         return create_geometry(geojson_data['type'], geojson_data['coordinates'])
 65 |     else:
 66 |         raise ValueError(f"Unsupported GeoJSON type: {geojson_data['type']}")
 67 | 
 68 | def parse_args():
 69 |     parser = argparse.ArgumentParser(description='Canopy Height Mapping using Earth Engine')
 70 |     # Basic parameters
 71 |     parser.add_argument('--aoi', type=str, required=True, help='Path to AOI GeoJSON file')
 72 |     parser.add_argument('--year', type=int, required=True, help='Year for analysis')
 73 |     parser.add_argument('--start-date', type=str, default='01-01', help='Start date (MM-DD)')
 74 |     parser.add_argument('--end-date', type=str, default='12-31', help='End date (MM-DD)')
 75 |     parser.add_argument('--clouds-th', type=float, default=65, help='Cloud threshold')
 76 |     parser.add_argument('--scale', type=int, default=30, help='Output resolution in meters')
 77 |     parser.add_argument('--mask-type', type=str, default='NDVI',
 78 |                        choices=['DW', 'FNF', 'NDVI', 'WC', 'CHM', 'ALL', 'none'],
 79 |                        help='Type of forest mask to apply')
 80 |     # resample method
 81 |     parser.add_argument('--resample', type=str, default='bilinear', choices=['bilinear', 'bicubic'],
 82 |                        help='Resampling method for image export')
 83 |     # ndvi threshold
 84 |     parser.add_argument('--ndvi-threshold', type=float, default=0.3, help='NDVI threshold for forest mask')
 85 |     
 86 |     # GEDI parameters
 87 |     parser.add_argument('--gedi-start-date', type=str, help='GEDI start date (YYYY-MM-DD)')
 88 |     parser.add_argument('--gedi-end-date', type=str, help='GEDI end date (YYYY-MM-DD)')
 89 |     parser.add_argument('--quantile', type=str, default='098', help='GEDI height quantile')
 90 |     parser.add_argument('--gedi-type', type=str, default='singleGEDI', help='GEDI data type')
 91 |     # Add buffer for AOI with default value 1000m
 92 |     parser.add_argument('--buffer', type=int, default=1000, help='Buffer size in meters')
 93 | 
 94 |     # Model parameters
 95 |     parser.add_argument('--model', type=str, default='RF', choices=['RF', 'GBM', 'CART'],
 96 |                        help='Machine learning model type')
 97 |     parser.add_argument('--num-trees-rf', type=int, default=100,
 98 |                        help='Number of trees for Random Forest')
 99 |     parser.add_argument('--min-leaf-pop-rf', type=int, default=1,
100 |                        help='Minimum leaf population for Random Forest')
101 |     parser.add_argument('--bag-frac-rf', type=float, default=0.5,
102 |                        help='Bagging fraction for Random Forest')
103 |     parser.add_argument('--max-nodes-rf', type=int, default=None,
104 |                        help='Maximum nodes for Random Forest')
105 |     
106 |     # Output parameters
107 |     parser.add_argument('--output-dir', type=str, default='outputs',
108 |                        help='Output directory for CSV and TIF files')
109 |     parser.add_argument('--export-training', action='store_true',
110 |                        help='Export training data as CSV')
111 |     # Export merged stack image as TIF
112 |     parser.add_argument('--export-stack', action='store_true',
113 |                        help='Export merged stack image as TIF')
114 |     # Export forest mask as TIF
115 |     parser.add_argument('--export-forest-mask', action='store_true',
116 |                        help='Export forest mask as TIF')
117 |     parser.add_argument('--export-predictions', action='store_true',
118 |                        help='Export predictions as TIF')
119 |     
120 |     args = parser.parse_args()
121 |     return args
122 | 
123 | def initialize_ee():
124 |     """Initialize Earth Engine with project ID."""
125 |     EE_PROJECT_ID = "my-project-423921"
126 |     ee.Initialize(project=EE_PROJECT_ID)
127 | 
128 | def export_training_data(reference_data: ee.FeatureCollection, output_dir: str):
129 |     """Export training data as CSV."""
130 |     # Get feature properties as a list of dictionaries
131 |     features = reference_data.getInfo()['features']
132 |     
133 |     # Extract properties and coordinates
134 |     data = []
135 |     for feature in features:
136 |         properties = feature['properties']
137 |         geometry = feature['geometry']['coordinates']
138 |         properties['longitude'] = geometry[0]
139 |         properties['latitude'] = geometry[1]
140 |         data.append(properties)
141 |     
142 |     # Convert to DataFrame and save
143 |     df = pd.DataFrame(data)
144 |     band_length = len(df.columns) - 3  # Exclude 'rh', 'longitude' and 'latitude'
145 |     df_size = len(df)
146 |     output_path = os.path.join(output_dir, f"training_data_b{band_length}_{df_size}.csv")
147 |     os.makedirs(output_dir, exist_ok=True)
148 |     df.to_csv(output_path, index=False)
149 |     print(f"Training data exported to: {output_path}")
150 |     
151 |     return output_path
152 | 
153 | def export_featurecollection_to_csv(feature_collection, export_name):
154 |     """Export a FeatureCollection to CSV via Earth Engine's batch export.
155 |     
156 |     Args:
157 |         feature_collection: The ee.FeatureCollection to export
158 |         export_name: Name for the exported file
159 |     """
160 |     # Add longitude and latitude columns
161 |     feature_collection = feature_collection.map(lambda feature: 
162 |         feature.set({
163 |             'longitude': feature.geometry().coordinates().get(0),
164 |             'latitude': feature.geometry().coordinates().get(1)
165 |         })
166 |     )
167 |     
168 |     # Get property names and convert to Python list
169 |     property_names = feature_collection.first().propertyNames().getInfo()
170 |     
171 |     # Remove system:index from the list if present
172 |     if 'system:index' in property_names:
173 |         property_names.remove('system:index')
174 |         
175 |     # Set up export task
176 |     export_task = ee.batch.Export.table.toDrive(
177 |         collection=feature_collection,
178 |         description=export_name,
179 |         fileNamePrefix=export_name,
180 |         folder='GEE_exports',  # Folder in your Google Drive
181 |         fileFormat='CSV',
182 |         selectors=property_names  # All property names
183 |     )
184 |     
185 |     # Start the export
186 |     export_task.start()
187 |     
188 |     print(f"Export started with task ID: {export_task.id}")
189 |     print("The CSV file will be available in your Google Drive once the export completes.")
190 | 
191 | def export_tif_via_ee(image: ee.Image, aoi: ee.Geometry, prefix: str, scale: int, resample: str = 'bilinear'):  
192 |     """Export predicted canopy height map as GeoTIFF using Earth Engine export."""
193 |     # Rename the classification band for clarity
194 |     # if 'classification' in image.bandNames().getInfo():
195 |     #     image = image.select(['classification'], ['canopy_height'])
196 |     band_count = image.bandNames().size().getInfo()
197 |     
198 |     pixel_area = ee.Image.pixelArea().divide(10000)  # Convert to hectares
199 |     area_img = ee.Image(1).rename('area').multiply(pixel_area)
200 |     
201 |     # Calculate total area in hectares
202 |     image_area_ha = area_img.reduceRegion(
203 |         reducer=ee.Reducer.sum(),
204 |         geometry=aoi,
205 |         scale=scale,
206 |         maxPixels=1e10
207 |     ).get('area')
208 |     
209 |     image_area_ha = int(round(image_area_ha.getInfo(), 0))
210 |     
211 |     # Generate a unique task ID (sanitize prefix and ensure valid characters)
212 |     clean_prefix = ''.join(c for c in prefix if c.isalnum() or c in '_-')
213 |     task_id = f"{clean_prefix}_b{band_count}_s{scale}_p{image_area_ha}"
214 |     
215 |     # image = image.resample(resample)  # or 'bicubic'
216 |     
217 |     # Set export parameters
218 |     export_params = {
219 |         'image': image,
220 |         'description': task_id,
221 |         'fileNamePrefix': task_id,
222 |         'folder': 'GEE_exports',
223 |         'scale': scale,
224 |         'region': aoi,
225 |         'fileFormat': 'GeoTIFF',
226 |         'maxPixels': 1e10
227 |     }
228 |     
229 |     # Start the export task
230 |     task = ee.batch.Export.image.toDrive(**export_params)
231 |     task.start()
232 |     
233 |     print(f"Export started with task ID: {task_id}")
234 |     print("The file will be available in your Google Drive once the export completes.")
235 |     
236 | def main():
237 |     """Main function to run the canopy height mapping process."""
238 |     # Parse arguments
239 |     args = parse_args()
240 |     
241 |     # Initialize Earth Engine
242 |     initialize_ee()
243 |     
244 |     # Load AOI
245 |     original_aoi = load_aoi(args.aoi)
246 |     aoi_buffered  = original_aoi.buffer(args.buffer)
247 |     
248 |     # Set dates
249 |     start_date = f"{args.year}-{args.start_date}"
250 |     end_date = f"{args.year}-{args.end_date}"
251 |     
252 |     # Create output directory
253 |     os.makedirs(args.output_dir, exist_ok=True)
254 |     
255 |     # Get S1, S2 satellite data
256 |     print("Collecting satellite data...")
257 |     s1 = get_sentinel1_data(aoi_buffered, args.year, args.start_date, args.end_date)
258 |     s2 = get_sentinel2_data(aoi_buffered, args.year, args.start_date, args.end_date, args.clouds_th)
259 |     # Import ALOS2 sar data
260 |     alos2 = get_alos2_data(aoi_buffered, args.year, args.start_date, args.end_date,include_texture=False,
261 |                 speckle_filter=False)
262 |     # Get terrain data
263 |     dem_data = get_dem_data(aoi_buffered)
264 |     
265 |     # Canopy height data
266 |     canopy_ht = get_canopyht_data(aoi_buffered)
267 |     
268 |     # Reproject datasets to the same projection
269 |     s2_projection = s2.projection()
270 |     s2 = s2.float()                                       # Convert to Float32
271 |     dem_data = dem_data.float()  # Convert to Float32
272 |     s1 = s1.float()              # Convert to Float32
273 |     alos2 = alos2.float()        # Convert to Float32
274 |     canopy_ht = canopy_ht.float()  # Convert to Float32
275 | 
276 |     # Merge datasets
277 |     merged = s2.addBands(s1).addBands(alos2).addBands(dem_data).addBands(canopy_ht)
278 |         
279 |     # Get predictor names before any masking
280 |     print("Getting band information...")
281 |     predictor_names = merged.bandNames()
282 |     n_predictors = predictor_names.size().getInfo()
283 |     var_split_rf = int(np.sqrt(n_predictors).round())
284 |         
285 |     # Create and apply forest mask
286 |     print(f"Creating and applying forest mask (type: {args.mask_type})...")
287 |     buffered_forest_mask = create_forest_mask(args.mask_type, aoi_buffered,
288 |                                    ee.Date(f"{args.year}-{args.start_date}"),
289 |                                    ee.Date(f"{args.year}-{args.end_date}"),
290 |                                    args.ndvi_threshold)
291 | 
292 |     forest_mask = buffered_forest_mask.clip(original_aoi)
293 |     
294 |     ndvi_threshold_percent = int(round(args.ndvi_threshold * 100,0))
295 |     
296 |     # Get GEDI data
297 |     print("Loading GEDI data...")
298 |     gedi = get_gedi_data(aoi_buffered, args.gedi_start_date, args.gedi_end_date, args.quantile)
299 |     
300 |     # forest_geometry = forest_mask.geometry()
301 |     # Sample GEDI points
302 |     gedi_points = gedi.sample(
303 |         region=aoi_buffered,
304 |         scale=args.scale,
305 |         geometries=True,
306 |         dropNulls=True,
307 |         seed=42
308 |     )
309 |     
310 |         # Sample points
311 |     reference_data = merged.sampleRegions(
312 |         collection=gedi_points,
313 |         scale=args.scale,
314 |         projection=s2_projection,
315 |         tileScale=1,
316 |         geometries=True
317 |     )
318 |     
319 |     # Clip to original AOI
320 |     merged = merged.clip(original_aoi)
321 |     
322 |     # Export training data if requested
323 |     if args.export_training:
324 |         print('Exporting training data and tif through Earth Engine...')
325 |         training_prefix = f'training_data_{args.mask_type}{ndvi_threshold_percent}_b{n_predictors}'
326 |         # export_training_data_via_ee(reference_data, training_prefix)
327 |         export_featurecollection_to_csv(reference_data, training_prefix)
328 |         print(f"Exporting training data as CSV: {training_prefix}.csv")
329 |     
330 |     if args.export_stack:
331 |         # Export the complete data stack
332 |         print("Exporting full data stack...")
333 |         export_tif_via_ee(merged, original_aoi, 'stack', args.scale, args.resample)
334 |         
335 |     if args.export_forest_mask:
336 |         # Export forest mask using export_tif_via_ee
337 |         forest_mask_prefix = f'forestMask{args.mask_type}{ndvi_threshold_percent}'
338 |         buffered_forest_mask_prefix = f'buffered_forestMask{args.mask_type}{ndvi_threshold_percent}'
339 |         # forest_mask_path = os.path.join(args.output_dir, f'{forest_mask_filename}.tif')
340 |         print(f"Exporting forest mask as {forest_mask_prefix}...")
341 |         export_tif_via_ee(forest_mask, original_aoi, forest_mask_prefix, args.scale, args.resample)
342 |         print(f"Exporting buffered forest mask as {buffered_forest_mask_prefix}...")
343 |         export_tif_via_ee(buffered_forest_mask, aoi_buffered, buffered_forest_mask_prefix, args.scale, args.resample)
344 | 
345 |     # Export predictions if requested
346 |     if args.export_predictions:
347 |         # Train model
348 |         print("Training model...")
349 |         if args.model == "RF":
350 |             classifier = ee.Classifier.smileRandomForest(
351 |                 numberOfTrees=args.num_trees_rf,
352 |                 variablesPerSplit=var_split_rf,
353 |                 minLeafPopulation=args.min_leaf_pop_rf,
354 |                 bagFraction=args.bag_frac_rf,
355 |                 maxNodes=args.max_nodes_rf
356 |             ).setOutputMode("Regression") \
357 |             .train(reference_data, "rh", predictor_names)
358 |         
359 |             # Generate predictions
360 |         print("Generating predictions...")
361 |         predictions = merged.classify(classifier)
362 |         # prediction_path = os.path.join(args.output_dir, 'predictions.tif')
363 |         print('Exporting via Earth Engine instead')
364 |         export_tif_via_ee(predictions, original_aoi, 'predictionCHM', args.scale, args.resample)
365 |     
366 |     print("Processing complete.")
367 | 
368 | if __name__ == "__main__":
369 |     main()


--------------------------------------------------------------------------------
/combine_heights.py:
--------------------------------------------------------------------------------
  1 | import os
  2 | import pandas as pd
  3 | import rasterio
  4 | from rasterio.warp import transform, reproject
  5 | import numpy as np
  6 | from utils import get_latest_file
  7 | from sklearn.metrics import mean_squared_error, mean_absolute_error
  8 | import matplotlib
  9 | matplotlib.use('Agg')  # Use non-interactive backend
 10 | import matplotlib.pyplot as plt
 11 | from scipy import stats
 12 | import json
 13 | 
 14 | def analyze_heights(df: pd.DataFrame, height_columns: list, ref_column: str = 'reference_height'):
 15 |     """
 16 |     Analyze relationships between reference heights and other height columns.
 17 |     """
 18 |     # Input validation
 19 |     if not isinstance(df, pd.DataFrame):
 20 |         raise TypeError("df must be a pandas DataFrame")
 21 |     if not isinstance(height_columns, list):
 22 |         raise TypeError("height_columns must be a list")
 23 |     if ref_column not in df.columns:
 24 |         raise ValueError(f"Reference column '{ref_column}' not found in DataFrame")
 25 |         
 26 |     # Filter out nodata and -32767 values from reference height
 27 |     valid_mask = (df[ref_column] != -32767) & df[ref_column].notna()
 28 |     valid_data = df[valid_mask].copy()
 29 |     
 30 |     print(f"\nHeight Analysis (using {len(valid_data)} valid points):")
 31 |     print("-" * 50)
 32 |     
 33 |     # Create a figure with subplots
 34 |     n_plots = len(height_columns)
 35 |     if n_plots > 0:
 36 |         n_cols = min(2, n_plots)
 37 |         n_rows = (n_plots + n_cols - 1) // n_cols
 38 |         fig = plt.figure(figsize=(12, 5*n_rows))
 39 |         axes = []
 40 |     else:
 41 |         print("No height columns to analyze")
 42 |         return None, None
 43 |     
 44 |     stats_dict = {}
 45 |     error_matrix = {}
 46 |     
 47 |     # Process each height column
 48 |     for idx, col in enumerate(height_columns):
 49 |         if col in valid_data.columns:
 50 |             print(f"\nAnalyzing {col}...")
 51 |             # Get valid pairs (no NaN in either column)
 52 |             valid_pairs = valid_data[[ref_column, col]].dropna()
 53 |             n_valid = len(valid_pairs)
 54 |             print(f"Found {n_valid} valid pairs for {col}")
 55 |             
 56 |             if n_valid < 2:
 57 |                 print(f"Skipping {col} - insufficient valid pairs")
 58 |                 continue
 59 |                 
 60 |             # Create subplot
 61 |             ax = fig.add_subplot(n_rows, n_cols, idx + 1)
 62 |             axes.append(ax)
 63 |             
 64 |             try:
 65 |                 # Convert to numpy arrays and ensure float type
 66 |                 ref_vals = valid_pairs[ref_column].values.astype(float)
 67 |                 col_vals = valid_pairs[col].values.astype(float)
 68 |                 
 69 |                 # Calculate statistics
 70 |                 rmse = float(np.sqrt(mean_squared_error(ref_vals, col_vals)))
 71 |                 mae = float(mean_absolute_error(ref_vals, col_vals))
 72 |                 bias = float(np.mean(col_vals - ref_vals))
 73 |                 correlation = float(np.corrcoef(ref_vals, col_vals)[0, 1])
 74 |                 
 75 |                 # Calculate error statistics
 76 |                 errors = col_vals - ref_vals
 77 |                 error_std = float(np.std(errors))
 78 |                 error_percentiles = [float(x) for x in np.percentile(errors, [5, 25, 50, 75, 95])]
 79 |                 
 80 |                 # Calculate regression
 81 |                 slope, intercept, r_value, p_value, std_err = stats.linregress(ref_vals, col_vals)
 82 |                 r_squared = r_value ** 2
 83 |                 
 84 |                 # Store statistics
 85 |                 stats_dict[col] = {
 86 |                     'N': n_valid,
 87 |                     'Correlation': correlation,
 88 |                     'RMSE': rmse,
 89 |                     'MAE': mae,
 90 |                     'Bias': bias,
 91 |                     'R-squared': r_squared,
 92 |                     'Slope': float(slope),
 93 |                     'Intercept': float(intercept),
 94 |                     'Error_std': error_std,
 95 |                     'Error_p5': error_percentiles[0],
 96 |                     'Error_p25': error_percentiles[1],
 97 |                     'Error_p50': error_percentiles[2],
 98 |                     'Error_p75': error_percentiles[3],
 99 |                     'Error_p95': error_percentiles[4]
100 |                 }
101 |                 error_matrix[col] = {'RMSE': rmse}
102 |                 
103 |                 # Print results
104 |                 print(f"Correlation: {correlation:.3f}")
105 |                 print(f"RMSE: {rmse:.3f} m")
106 |                 print(f"MAE: {mae:.3f} m")
107 |                 print(f"Bias: {bias:.3f} m")
108 |                 print(f"R-squared: {r_squared:.3f}")
109 |                 print(f"Linear fit: y = {slope:.3f}x + {intercept:.3f}")
110 |                 
111 |                 # Create scatter plot
112 |                 ax.scatter(ref_vals, col_vals, alpha=0.5, s=10)
113 |                 
114 |                 # Add identity line
115 |                 min_val = min(np.nanmin(ref_vals), np.nanmin(col_vals))
116 |                 max_val = max(np.nanmax(ref_vals), np.nanmax(col_vals))
117 |                 ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
118 |                 
119 |                 # Add regression line
120 |                 x_range = np.array([min_val, max_val])
121 |                 ax.plot(x_range, slope * x_range + intercept, 'g-', 
122 |                        label=f'Regression (R²={r_squared:.3f})')
123 |                 
124 |                 ax.set_xlabel(f'Reference Height (m)')
125 |                 ax.set_ylabel(f'{col} (m)')
126 |                 ax.set_title(f'Reference vs {col}\nRMSE={rmse:.2f}m, Bias={bias:.2f}m')
127 |                 ax.grid(True)
128 |                 ax.legend()
129 |                 
130 |             except Exception as e:
131 |                 print(f"Error analyzing {col}: {str(e)}")
132 |                 print(f"Data types - ref: {type(ref_vals)}, col: {type(col_vals)}")
133 |                 print(f"Data ranges - ref: [{np.nanmin(ref_vals)}, {np.nanmax(ref_vals)}], "
134 |                       f"col: [{np.nanmin(col_vals)}, {np.nanmax(col_vals)}]")
135 |                 continue
136 |     
137 |     stats_dict['error_matrix'] = error_matrix
138 |     
139 |     if len(axes) > 0:
140 |         plt.tight_layout()
141 |         return stats_dict, fig
142 |     else:
143 |         return None, None
144 | 
145 | def combine_heights_with_training(output_dir: str, reference_path: str):
146 |     """Combine reference heights with training data coordinates."""
147 |     # Input validation
148 |     if not os.path.exists(output_dir):
149 |         raise ValueError(f"Output directory does not exist: {output_dir}")
150 |     if not os.path.exists(reference_path):
151 |         raise ValueError(f"Reference file does not exist: {reference_path}")
152 |     
153 |     # Get latest training data file
154 |     training_file = get_latest_file(output_dir, 'training_data')
155 |     print(f"Using training file: {training_file}")
156 |     
157 |     # Read training data
158 |     df = pd.read_csv(training_file)
159 |     
160 |     # Open reference raster
161 |     with rasterio.open(reference_path) as src:
162 |         print(f"Reference CRS: {src.crs}")
163 |         
164 |         # Create lists of coordinates
165 |         lons = df['longitude'].values
166 |         lats = df['latitude'].values
167 |         
168 |         # Transform coordinates from EPSG:4326 to reference CRS
169 |         xs, ys = transform('EPSG:4326', src.crs, lons, lats)
170 |         
171 |         # Sample the raster at transformed coordinates
172 |         coords = list(zip(xs, ys))
173 |         samples = list(src.sample(coords))
174 |         heights = [sample[0] for sample in samples]
175 |     
176 |     # Create a Series with the heights
177 |     height_series = pd.Series(heights)
178 |     
179 |     # Replace -32767 with pd.NA
180 |     height_series = height_series.replace(-32767, pd.NA)
181 |     
182 |     # Add heights to dataframe
183 |     df['reference_height'] = height_series
184 |     
185 |     # Analyze relationships with other height columns
186 |     height_columns = ['rh', 'ch_potapov2021', 'ch_lang2022',
187 |                       'ch_pauls2024', 'ch_tolan2024']
188 |     
189 |     # Check column existence
190 |     available_columns = [col for col in height_columns if col in df.columns]
191 |     if not available_columns:
192 |         print("\nWarning: None of the specified height columns found in the data")
193 |         print(f"Available columns: {df.columns.tolist()}")
194 |     else:
195 |         print(f"\nAnalyzing height columns: {available_columns}")
196 |         stats, fig = analyze_heights(df, available_columns)
197 |         
198 |         if stats is not None:
199 |             # Save the statistics to a JSON file
200 |             stats_file = os.path.join(output_dir, 'trainingData_height_analysis.json')
201 |             with open(stats_file, 'w') as f:
202 |                 json.dump(stats, f, indent=4)
203 |             print(f"\nStatistics saved to: {stats_file}")
204 |             
205 |             # Save the analysis plot
206 |             plot_path = os.path.join(output_dir, 'trainingData_height_comparison.png')
207 |             fig.savefig(plot_path, dpi=300, bbox_inches='tight')
208 |             plt.close(fig)
209 |             print(f"Analysis plot saved to: {plot_path}")
210 |             
211 |             # Print error matrix
212 |             print("\nError Matrix (RMSE between reference and height columns):")
213 |             error_df = pd.DataFrame.from_dict({col: data['RMSE'] 
214 |                                    for col, data in stats['error_matrix'].items()}, 
215 |                                    orient='index', columns=['RMSE'])
216 |             print(error_df.round(3))
217 |     
218 |     # Save combined data with explicit NA handling
219 |     output_file = os.path.join(output_dir, 'trainingData_with_heights.csv')
220 |     df.to_csv(output_file, index=False, na_rep='NA')
221 |     
222 |     # Print summary statistics
223 |     valid_heights = df['reference_height'].dropna()
224 |     print(f"\nSummary:")
225 |     print(f"Total points: {len(df)}")
226 |     print(f"Valid heights: {len(valid_heights)}")
227 |     print(f"No data points: {len(df) - len(valid_heights)}")
228 |     if len(valid_heights) > 0:
229 |         print(f"Height range: {valid_heights.min():.2f} to {valid_heights.max():.2f}")
230 |     print(f"\nCombined data saved to: {output_file}")
231 | 
232 | if __name__ == "__main__":
233 |     # Use chm_outputs directory
234 |     output_dir = "chm_outputs"
235 |     reference = f"{output_dir}/dchm_09id4.tif"
236 |     combine_heights_with_training(output_dir, reference)


--------------------------------------------------------------------------------
/dem_source.py:
--------------------------------------------------------------------------------
 1 | import ee
 2 | 
 3 | 
 4 | def get_dem_data(aoi: ee.Geometry):
 5 |     try:
 6 |         print("Downloading GMTED data")
 7 |         dem = ee.Image("USGS/GMTED2010_FULL").select(['min'], ['elevation']) #"USGS/GMTED2010" was deprecated.
 8 |         slope = ee.Terrain.slope(dem)
 9 |         aspect = ee.Terrain.aspect(dem)
10 |         dem_data = dem.addBands(slope).addBands(aspect).select(['elevation', 'slope', 'aspect'], ['GMTED_elevation', 'GMTED_slope', 'GMTED_aspect']).clip(aoi)
11 |     except Exception as e:
12 |         print(f"Error loading GMTED data: {e}")
13 |         dem_data = None
14 |     try:
15 |         # print(f"Downloading SRTM data instead{}")
16 |         print("Donwloading SRTM data")
17 |         dem = ee.Image("USGS/SRTMGL1_003").select('elevation')
18 |         slope = ee.Terrain.slope(dem)
19 |         aspect = ee.Terrain.aspect(dem)
20 |         dem_data_SRTM = dem.addBands(slope).addBands(aspect).select(['elevation', 'slope', 'aspect'], ['SRTM_elevation', 'SRTM_slope', 'SRTM_aspect']).clip(aoi)
21 |         dem_data = dem_data.addBands(dem_data_SRTM)
22 |     except Exception as e:
23 |         print(f"Error loading SRTM data: {e}")
24 |     try:    
25 |         print("Downloading ALOS AW3D30 data")
26 |         dem = ee.ImageCollection("JAXA/ALOS/AW3D30/V3_2").mosaic().select('DSM')
27 |         # proj = dem.select(0).projection()
28 |         # slope = ee.Terrain.slope(dem.setDefaultProjection(proj)) # does not work
29 |         # aspect = ee.Terrain.aspect(dem.setDefaultProjection(proj)) # does not work
30 |         # slope = ee.Terrain.slope(dem)
31 |         # aspect = ee.Terrain.aspect(dem)
32 |         # dem_data_AW3D30 = dem.addBands(slope).addBands(aspect).select(['DSM', 'slope', 'aspect'], ['AW3D30_elevation', 'AW3D30_slope', 'AW3D30_aspect']).clip(aoi)
33 |         dem_data_AW3D30 = dem.rename(['AW3D30_elevation'])
34 |         dem_data = dem_data.addBands(dem_data_AW3D30)
35 |     except Exception as e:
36 |         print(f"Error loading ALOS AW3D30 data: {e}")
37 |     
38 |     try:
39 |         print("Downloading GLO30 data")
40 |         dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").mosaic().select('DEM')
41 |         slope = ee.Terrain.slope(dem)
42 |         aspect = ee.Terrain.aspect(dem)
43 |         dem_data_GLO30 = dem.addBands(slope).addBands(aspect).select(['DEM', 'slope', 'aspect'], ['GLO30_elevation', 'GLO30_slope', 'GLO30_aspect']).clip(aoi)
44 |         dem_data = dem_data.addBands(dem_data_GLO30)
45 |     except Exception as e:
46 |         print(f"Error loading GLO30 data: {e}")
47 |         
48 |     return dem_data
49 | 


--------------------------------------------------------------------------------
/dl_models.py:
--------------------------------------------------------------------------------
  1 | import torch
  2 | import torch.nn as nn
  3 | import torch.nn.functional as F
  4 | from torch import optim
  5 | from torch.utils.data import DataLoader, TensorDataset
  6 | import pandas as pd
  7 | import numpy as np
  8 | 
  9 | def create_normalized_dataloader(X_train, X_val, y_train, y_val, batch_size=64, n_bands=None):
 10 |     """
 11 |     Create normalized dataloaders for training and validation data with band-by-band normalization.
 12 |     
 13 |     Args:
 14 |         X_train: Training features
 15 |         X_val: Validation features
 16 |         y_train: Training targets
 17 |         y_val: Validation targets
 18 |         batch_size: Batch size for dataloaders
 19 |         n_bands: Number of spectral bands per feature. If None, treats each feature independently.
 20 |         
 21 |     Returns:
 22 |         train_loader: Normalized training dataloader
 23 |         val_loader: Normalized validation dataloader
 24 |         scaler_mean: Feature means for denormalization (shape: n_features)
 25 |         scaler_std: Feature standard deviations for denormalization (shape: n_features)
 26 |     """
 27 |     # Convert to torch tensors
 28 |     X_train_tensor = torch.FloatTensor(X_train)
 29 |     X_val_tensor = torch.FloatTensor(X_val)
 30 |     y_train_tensor = torch.FloatTensor(y_train)
 31 |     y_val_tensor = torch.FloatTensor(y_val)
 32 |     
 33 |     # If n_bands is provided, reshape tensors to group by bands
 34 |     if n_bands is not None:
 35 |         n_train_samples = X_train_tensor.shape[0]
 36 |         n_val_samples = X_val_tensor.shape[0]
 37 |         n_features = X_train_tensor.shape[1]
 38 |         n_groups = n_features // n_bands
 39 |         
 40 |         # Reshape to (samples, n_groups, n_bands)
 41 |         X_train_tensor = X_train_tensor.view(n_train_samples, n_groups, n_bands)
 42 |         X_val_tensor = X_val_tensor.view(n_val_samples, n_groups, n_bands)
 43 |         
 44 |         # Calculate normalization parameters for each band within each group
 45 |         scaler_mean = X_train_tensor.mean(dim=0)  # Shape: (n_groups, n_bands)
 46 |         scaler_std = X_train_tensor.std(dim=0)    # Shape: (n_groups, n_bands)
 47 |         scaler_std[scaler_std == 0] = 1  # Prevent division by zero
 48 |         
 49 |         # Normalize band by band
 50 |         X_train_normalized = (X_train_tensor - scaler_mean) / scaler_std
 51 |         X_val_normalized = (X_val_tensor - scaler_mean) / scaler_std
 52 |         
 53 |         # Reshape back to original format
 54 |         X_train_normalized = X_train_normalized.reshape(n_train_samples, -1)
 55 |         X_val_normalized = X_val_normalized.reshape(n_val_samples, -1)
 56 |         scaler_mean = scaler_mean.reshape(-1)
 57 |         scaler_std = scaler_std.reshape(-1)
 58 |     else:
 59 |         # Calculate normalization parameters for each feature independently
 60 |         scaler_mean = X_train_tensor.mean(dim=0)
 61 |         scaler_std = X_train_tensor.std(dim=0)
 62 |         scaler_std[scaler_std == 0] = 1  # Prevent division by zero
 63 |         
 64 |         # Normalize data
 65 |         X_train_normalized = (X_train_tensor - scaler_mean) / scaler_std
 66 |         X_val_normalized = (X_val_tensor - scaler_mean) / scaler_std
 67 |     
 68 |     # Create TensorDatasets
 69 |     train_dataset = TensorDataset(X_train_normalized, y_train_tensor)
 70 |     val_dataset = TensorDataset(X_val_normalized, y_val_tensor)
 71 |     
 72 |     # Create DataLoaders
 73 |     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
 74 |     val_loader = DataLoader(val_dataset, batch_size=batch_size)
 75 |     
 76 |     return train_loader, val_loader, scaler_mean, scaler_std
 77 | 
 78 | class MLPRegressionModel(torch.nn.Module):
 79 |     def __init__(self, input_size: int, num_layers: int = 3, nodes: int = 1024,
 80 |                 dropout: float = 0.2, is_nodes_half: bool = False):
 81 |         super().__init__()
 82 |         self.num_features = input_size
 83 |         self.layers = nn.ModuleList()
 84 |         self.batch_norms = nn.ModuleList()
 85 |         self.activation = nn.ReLU()
 86 | 
 87 |         self.layers.append(nn.Linear(self.num_features, nodes))
 88 |         self.batch_norms.append(nn.BatchNorm1d(nodes))
 89 |         
 90 |         self.dropout = nn.Dropout(dropout)
 91 |         
 92 |         if is_nodes_half:
 93 |             for i in range(num_layers):
 94 |                 in_features = int(nodes / (i+1))
 95 |                 out_features = int(nodes / (i+2))
 96 |                 self.layers.append(nn.Linear(in_features, out_features))
 97 |                 self.batch_norms.append(nn.BatchNorm1d(out_features))
 98 |             self.head = nn.Linear(int(nodes / (num_layers+1)), 1)
 99 |         else:
100 |             for _ in range(num_layers):
101 |                 self.layers.append(nn.Linear(nodes, nodes))
102 |                 self.batch_norms.append(nn.BatchNorm1d(nodes))
103 |             self.head = nn.Linear(nodes, 1)
104 |     
105 |     def forward(self, x):
106 |         for layer, bn in zip(self.layers, self.batch_norms):
107 |             x = layer(x)
108 |             x = bn(x)
109 |             x = self.activation(x)
110 |             x = self.dropout(x)
111 |         
112 |         x = self.head(x)
113 |         return x.squeeze(1)
114 | 


--------------------------------------------------------------------------------
/evaluate_predictions.py:
--------------------------------------------------------------------------------
  1 | """Module for evaluating canopy height predictions."""
  2 | 
  3 | import os
  4 | import numpy as np
  5 | import argparse
  6 | from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
  7 | import rasterio
  8 | from datetime import datetime
  9 | from rasterio.crs import CRS
 10 | from rasterio.warp import transform_bounds
 11 | 
 12 | from save_evaluation_pdf import save_evaluation_to_pdf
 13 | from raster_utils import load_and_align_rasters
 14 | from evaluation_utils import validate_data, create_plots
 15 | 
 16 | 
 17 | def check_predictions(pred_path: str):
 18 |     """Check if predictions are valid before proceeding."""
 19 |     with rasterio.open(pred_path) as src:
 20 |         data = src.read(1)
 21 |         if np.all(data == src.nodata):
 22 |             print(f"\nError: The prediction file {os.path.basename(pred_path)} contains only nodata values.")
 23 |             print("Please ensure the prediction generation completed successfully.")
 24 |             return False
 25 |         return True
 26 | 
 27 | 
 28 | def calculate_metrics(pred: np.ndarray, ref: np.ndarray)->dict:
 29 |     """Calculate evaluation metrics."""
 30 |     mse = mean_squared_error(ref, pred)
 31 |     rmse = np.sqrt(mse)
 32 |     mae = mean_absolute_error(ref, pred)
 33 |     r2 = r2_score(ref, pred)
 34 |     
 35 |     # Calculate additional statistics
 36 |     errors = pred - ref
 37 |     mean_error = np.mean(errors)
 38 |     std_error = np.std(errors)
 39 |     max_error = np.max(np.abs(errors))
 40 |     
 41 |     # Calculate percentage of predictions within different error ranges
 42 |     within_1m = np.mean(np.abs(errors) <= 1.0) * 100
 43 |     within_2m = np.mean(np.abs(errors) <= 2.0) * 100
 44 |     within_5m = np.mean(np.abs(errors) <= 5.0) * 100
 45 |     
 46 |     return {
 47 |         'MSE': mse,
 48 |         'RMSE': rmse,
 49 |         'MAE': mae,
 50 |         'R2': r2,
 51 |         'Mean Error': mean_error,
 52 |         'Std Error': std_error,
 53 |         'Max Absolute Error': max_error,
 54 |         'Within 1m (%)': within_1m,
 55 |         'Within 2m (%)': within_2m,
 56 |         'Within 5m (%)': within_5m
 57 |     }
 58 | 
 59 | 
 60 | def main():
 61 |     # Parse command line arguments
 62 |     parser = argparse.ArgumentParser(description='Evaluate canopy height predictions against reference data')
 63 |     parser.add_argument('--pred', type=str, help='Path to prediction raster', default='chm_outputs/predictions.tif')
 64 |     parser.add_argument('--ref', type=str, help='Path to reference raster', default='chm_outputs/dchm_09id4.tif')
 65 |     parser.add_argument('--forest-mask', type=str, help='Path to forest mask raster', default=None)
 66 |     parser.add_argument('--output', type=str, help='Output directory', default='chm_outputs/evaluation')
 67 |     parser.add_argument('--pdf', action='store_true', help='Generate PDF report with 2x2 comparison grid')
 68 |     parser.add_argument('--model-eval', type=str, help='Path to model evaluation JSON file', default=None)
 69 |     parser.add_argument('--training', type=str, help='Path to training data CSV for additional metadata', default='chm_outputs/training_data.csv')
 70 |     parser.add_argument('--merged', type=str, help='Path to merged data raster for RGB visualization', default=None)
 71 |     args = parser.parse_args()
 72 |     
 73 |     # Set paths
 74 |     pred_path = args.pred
 75 |     ref_path = args.ref
 76 |     output_dir = args.output
 77 |     generate_pdf = args.pdf
 78 |     training_data_path = args.training if os.path.exists(args.training) else None
 79 |     merged_data_path = args.merged if args.merged and os.path.exists(args.merged) else None
 80 |     
 81 |     # Create output directory
 82 |     os.makedirs(output_dir, exist_ok=True)
 83 |     # Add date to output directory
 84 |     date = datetime.now().strftime("%Y%m%d")
 85 |     output_dir = os.path.join(output_dir, date)
 86 |     os.makedirs(output_dir, exist_ok=True)
 87 |     
 88 |     # First check if predictions are valid
 89 |     if not check_predictions(pred_path):
 90 |         return 1
 91 | 
 92 |     try:
 93 |         
 94 |         print("Loading and preprocessing rasters...")
 95 |         pred_data, ref_data, transform, forest_mask = load_and_align_rasters(
 96 |             pred_path, ref_path, args.forest_mask, output_dir)
 97 |         
 98 |         # Create masks for no data values and outliers
 99 |         print("\nCreating valid data masks...")
100 |         pred_mask = (pred_data >= 0) & (pred_data <= 35) & ~np.isnan(pred_data)  # Reasonable height range for trees
101 |         ref_mask = (ref_data >= 0) & (ref_data <= 35) & (ref_data != -32767) & ~np.isnan(ref_data)  # Exclude -32767 and no data      # Same range for reference
102 |         
103 |         # Combine all masks
104 |         mask = pred_mask & ref_mask
105 |         if forest_mask is not None:
106 |             mask = mask & forest_mask
107 |             print(f"Applied forest mask - {np.sum(forest_mask):,} forest pixels")
108 |         
109 |         valid_pixels = np.sum(mask)
110 |         total_pixels = mask.size
111 |         print(f"Valid pixels: {valid_pixels:,} of {total_pixels:,} ({valid_pixels/total_pixels*100:.1f}%)")
112 |         # Calculate area using geographic coordinates
113 |         with rasterio.open(pred_path) as src:
114 |             # Get the CRS of the prediction
115 |             if src.crs.is_geographic:
116 |                 # For geographic coordinates, calculate approximate area using UTM
117 |                 center_lat = (src.bounds.bottom + src.bounds.top) / 2
118 |                 center_lon = (src.bounds.left + src.bounds.right) / 2
119 |                 utm_zone = int((center_lon + 180) / 6) + 1
120 |                 utm_epsg = 32600 + utm_zone + (0 if center_lat >= 0 else 100)
121 |                 utm_crs = CRS.from_epsg(utm_epsg)
122 |                 
123 |                 # Transform bounds to UTM
124 |                 bounds_utm = transform_bounds(src.crs, utm_crs, *src.bounds)
125 |                 width_m = bounds_utm[2] - bounds_utm[0]
126 |                 height_m = bounds_utm[3] - bounds_utm[1]
127 |                 
128 |                 # Calculate pixel size in meters
129 |                 pixel_width_m = width_m / src.width
130 |                 pixel_height_m = height_m / src.height
131 |                 pixel_area_m2 = pixel_width_m * pixel_height_m
132 |             else:
133 |                 # For projected coordinates, use transform directly
134 |                 pixel_area_m2 = abs(transform[0] * transform[4])
135 |         
136 |         area_ha = (np.sum(mask) * pixel_area_m2) / 10000  # Convert to hectares
137 |         print(f"Area of valid pixels: {area_ha:.2f} ha")
138 |         
139 |         if valid_pixels == 0:
140 |             raise ValueError("No valid pixels in intersection area")
141 |         
142 |         # ref_data = ref_data[mask]
143 |         # pred_data = pred_data[mask]
144 |         
145 |         if mask is not None:
146 |             ref_masked_2d = np.ma.masked_where(~mask, ref_data)
147 |             pred_masked_2d = np.ma.masked_where(~mask, pred_data)
148 |         else:
149 |             ref_masked_2d = ref_data
150 |             pred_masked_2d = pred_data
151 |         
152 |         ref_masked_2 = ref_masked_2d.compressed() 
153 |         pred_masked_2 = pred_masked_2d.compressed() 
154 |         
155 |         ref_masked = ref_data[mask]
156 |         pred_masked = pred_data[mask]
157 | 
158 |         # Print statistics
159 |         print("\nStatistics for valid pixels (filtered to 0-35m range, excluding -32767 and no data):")
160 |         print("Prediction - Min: {:.2f}, Max: {:.2f}, Mean: {:.2f}, Std: {:.2f}".format(
161 |             np.min(pred_masked), np.max(pred_masked), np.mean(pred_masked), np.std(pred_masked)))
162 |         print("Reference - Min: {:.2f}, Max: {:.2f}, Mean: {:.2f}, Std: {:.2f}".format(
163 |             np.min(ref_masked), np.max(ref_masked), np.mean(ref_masked), np.std(ref_masked)))
164 |         
165 |         # Validate data and get statistics
166 |         print("\nValidating data...")
167 |         # validation_info = validate_data(pred_masked, ref_masked)
168 |         validation_info = validate_data(pred_masked_2, ref_masked_2)
169 |         print("Data validation passed:")
170 |         print(f"Prediction range: {validation_info['pred_range'][0]:.2f} to {validation_info['pred_range'][1]:.2f}")
171 |         print(f"Reference range: {validation_info['ref_range'][0]:.2f} to {validation_info['ref_range'][1]:.2f}")
172 |         
173 |         # Calculate metrics
174 |         print("Calculating metrics...")
175 |         # metrics = calculate_metrics(pred_masked, ref_masked)
176 |         metrics = calculate_metrics(pred_masked_2, ref_masked_2)
177 |         
178 |         print("Generating visualizations...")
179 |         # Always generate plots for masked data
180 |         # plot_paths = create_plots(pred_masked, ref_masked, metrics, output_dir)
181 |         plot_paths = create_plots(pred_masked_2, ref_masked_2, metrics, output_dir)
182 |         
183 |         if generate_pdf:
184 |             # Create PDF report with all visualizations
185 |             print("\nGenerating PDF report...")
186 |             
187 |             # If model evaluation path is not provided, look for it in the parent directory
188 |             if args.model_eval is None:
189 |                 model_eval_path = os.path.join(os.path.dirname(output_dir), 'model_evaluation.json')
190 |             else:
191 |                 model_eval_path = args.model_eval
192 |                 
193 |             if os.path.exists(model_eval_path):
194 |                 print(f"Including model evaluation data from: {model_eval_path}")
195 |             
196 |             pdf_path = save_evaluation_to_pdf(
197 |                 pred_path,
198 |                 ref_path,
199 |                 pred_data,
200 |                 ref_data,
201 |                 metrics,
202 |                 output_dir,
203 |                 mask=mask,
204 |                 training_data_path=training_data_path,
205 |                 merged_data_path=merged_data_path,
206 |                 area_ha=area_ha,
207 |                 validation_info=validation_info,
208 |                 plot_paths=plot_paths
209 |             )
210 |             print(f"PDF report saved to: {pdf_path}")
211 |         
212 |         # Print results
213 |         print("\nEvaluation Results (for heights between 0-35m, excluding -32767 and no data):")
214 |         print("-" * 50)
215 |         for metric, value in metrics.items():
216 |             if metric.endswith('(%)'):
217 |                 print(f"{metric:<20}: {value:>7.1f}%")
218 |             else:
219 |                 print(f"{metric:<20}: {value:>7.3f}")
220 |         print("-" * 50)
221 |         
222 |         print("\nOutputs saved to:", output_dir)
223 |         
224 |         return 0
225 |         
226 |     except ValueError as e:
227 |         print(f"\nValidation Error: {str(e)}")
228 |         print("\nPlease check that both rasters contain valid height values.")
229 |         return 1
230 |     except Exception as e:
231 |         print(f"\nError: {str(e)}")
232 |         return 1
233 | 
234 | 
235 | if __name__ == "__main__":
236 |     import sys
237 |     sys.exit(main())


--------------------------------------------------------------------------------
/evaluation_utils.py:
--------------------------------------------------------------------------------
  1 | """Shared evaluation utilities."""
  2 | 
  3 | import os
  4 | import numpy as np
  5 | import matplotlib.pyplot as plt
  6 | from scipy.stats import norm
  7 | 
  8 | 
  9 | def validate_data(pred_data: np.ndarray, ref_data: np.ndarray):
 10 |     """Validate data before analysis and return validation info."""
 11 |     validation_info = {
 12 |         'pred_range': (np.min(pred_data), np.max(pred_data)),
 13 |         'ref_range': (np.min(ref_data), np.max(ref_data)),
 14 |         'pred_stats': {'mean': np.mean(pred_data), 'std': np.std(pred_data)},
 15 |         'ref_stats': {'mean': np.mean(ref_data), 'std': np.std(ref_data)}
 16 |     }
 17 |     
 18 |     # Check for zero variance
 19 |     pred_std = validation_info['pred_stats']['std']
 20 |     if pred_std == 0:
 21 |         raise ValueError("Prediction data has zero variance (all values are the same). " +
 22 |                         f"All values are {pred_data[0]:.2f}")
 23 |     
 24 |     ref_std = validation_info['ref_stats']['std']
 25 |     if ref_std == 0:
 26 |         raise ValueError("Reference data has zero variance (all values are the same). " +
 27 |                         f"All values are {ref_data[0]:.2f}")
 28 |     
 29 |     # Check for reasonable value ranges
 30 |     if validation_info['pred_range'][1] < 0.01:
 31 |         raise ValueError(f"Prediction values seem too low. Max value is {validation_info['pred_range'][1]:.6f}")
 32 |     
 33 |     if validation_info['ref_range'][1] < 0.01:
 34 |         raise ValueError(f"Reference values seem too low. Max value is {validation_info['ref_range'][1]:.6f}")
 35 |     
 36 |     return validation_info
 37 | 
 38 | 
 39 | def create_plots(pred: np.ndarray, ref: np.ndarray, metrics: dict, output_dir: str):
 40 |     """Create evaluation plots and return plot paths."""
 41 |     plot_paths = {}
 42 |     
 43 |     # Scatter plot
 44 |     plt.figure(figsize=(10, 10))
 45 |     plt.scatter(ref, pred, alpha=0.5, s=1)
 46 |     plt.plot([0, max(ref.max(), pred.max())], [0, max(ref.max(), pred.max())], 'r--', label='1:1 line')
 47 |     
 48 |     # Add trend line
 49 |     z = np.polyfit(ref, pred, 1)
 50 |     p = np.poly1d(z)
 51 |     plt.plot(ref, p(ref), 'b--', label=f'Trend line (y = {z[0]:.3f}x + {z[1]:.3f})')
 52 |     
 53 |     plt.xlabel('Reference Height (m)')
 54 |     plt.ylabel('Predicted Height (m)')
 55 |     plt.title('Predicted vs Reference Height\n' + \
 56 |              f'R² = {metrics["R2"]:.3f}, RMSE = {metrics["RMSE"]:.3f}m')
 57 |     plt.legend()
 58 |     plt.grid(True)
 59 |     plot_paths['scatter'] = os.path.join(output_dir, 'scatter_plot.png')
 60 |     plt.savefig(plot_paths['scatter'], dpi=300, bbox_inches='tight')
 61 |     plt.close()
 62 |     
 63 |     # Error histogram
 64 |     errors = pred - ref
 65 |     plt.figure(figsize=(10, 6))
 66 |     plt.hist(errors, bins=50, alpha=0.75, density=True)
 67 |     plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
 68 |     
 69 |     # Add normal distribution curve
 70 |     xmin, xmax = plt.xlim()
 71 |     x = np.linspace(xmin, xmax, 100)
 72 |     p = norm.pdf(x, errors.mean(), errors.std())
 73 |     plt.plot(x, p, 'k--', label='Normal Distribution')
 74 |     
 75 |     plt.xlabel('Prediction Error (m)')
 76 |     plt.ylabel('Density')
 77 |     plt.title(f'Error Distribution\n' + \
 78 |              f'Mean = {errors.mean():.3f}m, Std = {errors.std():.3f}m')
 79 |     plt.legend()
 80 |     plt.grid(True)
 81 |     plot_paths['error_hist'] = os.path.join(output_dir, 'error_hist.png')
 82 |     plt.savefig(plot_paths['error_hist'], dpi=300, bbox_inches='tight')
 83 |     plt.close()
 84 |     
 85 |     # Height distributions
 86 |     plt.figure(figsize=(10, 6))
 87 |     plt.hist(ref, bins=50, alpha=0.5, label='Reference', density=True)
 88 |     plt.hist(pred, bins=50, alpha=0.5, label='Predicted', density=True)
 89 |     plt.xlabel('Height (m)')
 90 |     plt.ylabel('Density')
 91 |     plt.title('Height Distributions')
 92 |     plt.legend()
 93 |     plt.grid(True)
 94 |     plot_paths['height_dist'] = os.path.join(output_dir, 'height_distributions.png')
 95 |     plt.savefig(plot_paths['height_dist'], dpi=300, bbox_inches='tight')
 96 |     plt.close()
 97 |     
 98 |     return plot_paths
 99 | 
100 | 
101 | def create_comparison_grid(ref_data, pred_data, diff_data, rgb_data, output_path, forest_mask=None):
102 |     """Create 2x2 grid visualization and save to file."""
103 |     fig, axes = plt.subplots(2, 2, figsize=(10, 10))
104 |     vmax = 35
105 |     # Create masked versions for visualization
106 |     if forest_mask is not None:
107 |         ref_masked = np.ma.masked_where(~forest_mask, ref_data)
108 |         pred_masked = np.ma.masked_where(~forest_mask, pred_data)
109 |         diff_masked = np.ma.masked_where(~forest_mask, diff_data)
110 |     else:
111 |         ref_masked = ref_data
112 |         pred_masked = pred_data
113 |         diff_masked = diff_data
114 |     
115 |     # Plot reference data
116 |     im0 = axes[0,0].imshow(ref_masked, cmap='viridis', vmin=0, vmax=vmax, aspect='equal')
117 |     axes[0,0].set_title('Reference Heights')
118 |     plt.colorbar(im0, ax=axes[0,0], fraction=0.046, pad=0.04)
119 |     
120 |     # Plot prediction data
121 |     im1 = axes[0,1].imshow(pred_masked, cmap='viridis', vmin=0, vmax=vmax, aspect='equal')
122 |     axes[0,1].set_title('Predicted Heights')
123 |     plt.colorbar(im1, ax=axes[0,1], fraction=0.046, pad=0.04)
124 |     
125 |     # Plot difference map
126 |     im2 = axes[1,0].imshow(diff_masked, cmap='RdYlBu', vmin=-10, vmax=10, aspect='equal')
127 |     axes[1,0].set_title('Height Difference (Pred - Ref)')
128 |     plt.colorbar(im2, ax=axes[1,0], fraction=0.046, pad=0.04)
129 |     
130 |     
131 |     # Plot RGB or empty plot
132 |     if rgb_data is not None:
133 |         axes[1,1].imshow(rgb_data, aspect='equal')
134 |         axes[1,1].set_title('RGB Composite')
135 |         # if forest_mask is not None:
136 |     #         # Apply forest mask to RGB data
137 |     #         rgb_masked = rgb_data.copy()
138 |     #         for i in range(3):
139 |     #             rgb_masked[:,:,i] = np.where(forest_mask, rgb_data[:,:,i], 0)
140 |             # axes[1,1].imshow(rgb_masked, aspect='equal')
141 |     #     else:
142 |     #         axes[1,1].imshow(rgb_data, aspect='equal')
143 |     #     axes[1,1].set_title('RGB Composite')
144 |     else:
145 |         axes[1,1].imshow(np.zeros_like(pred_data), cmap='gray', aspect='equal')
146 |         axes[1,1].set_title('RGB Not Available')
147 |     
148 |     # Remove axes ticks
149 |     for ax in axes.flat:
150 |         ax.set_xticks([])
151 |         ax.set_yticks([])
152 |     
153 |     plt.tight_layout()
154 |     plt.savefig(output_path, dpi=300, bbox_inches='tight')
155 |     plt.close()


--------------------------------------------------------------------------------
/for_forest_masking.py:
--------------------------------------------------------------------------------
  1 | import ee
  2 | # import geopandas as gpd
  3 | import pandas as pd
  4 | from typing import Union, List, Tuple
  5 | 
  6 | def create_forest_mask(
  7 |     mask_type: str,
  8 |     aoi: ee.Geometry,
  9 |     start_date_ee: ee.Date,
 10 |     end_date_ee: ee.Date,
 11 |     ndvi_threshold: float = 0.3
 12 | ) -> ee.Image:
 13 |     """
 14 |     Create a forest mask based on specified mask type.
 15 |     
 16 |     Args:
 17 |         mask_type: Type of mask to create ('DW', 'FNF', 'NDVI', 'ALL', 'none')
 18 |         aoi: Area of interest as Earth Engine Geometry
 19 |         start_date_ee: Start date as ee.Date
 20 |         end_date_ee: End date as ee.Date
 21 |         ndvi_threshold: NDVI threshold for forest classification (default: 0.2)
 22 |     
 23 |     Returns:
 24 |         ee.Image: Binary forest mask (1 for forest, 0 for non-forest)
 25 |     """
 26 |     # Initialize masks with default (all ones)
 27 |     # dw_mask = ee.Image(1).clip(aoi)
 28 |     # fnf_mask = ee.Image(1).clip(aoi)
 29 |     # ndvi_mask = ee.Image(1).clip(aoi)
 30 |     dw_mask = ee.Image(0).clip(aoi)
 31 |     fnf_mask = ee.Image(0).clip(aoi)
 32 |     ndvi_mask = ee.Image(0).clip(aoi)
 33 |     wc_mask = ee.Image(0).clip(aoi)
 34 |     ch_mask = ee.Image(0).clip(aoi)
 35 |     
 36 |     # Create a buffered version of the AOI to ensure we get all relevant tiles
 37 |     buffered_aoi = aoi.buffer(10000)  # Buffer by 5km
 38 |     
 39 |     def clip_image(image):
 40 |         """Clip image to the AOI."""
 41 |         return image.clip(aoi)
 42 |     # Create Dynamic World mask if requested
 43 |     if mask_type in ['DW', 'ALL']:
 44 |         # Import Dynamic World dataset using buffered AOI
 45 |         dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
 46 |             .filterBounds(buffered_aoi) \
 47 |             .filterDate(start_date_ee, end_date_ee)
 48 |         # dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
 49 |         #     .filterDate(start_date_ee, end_date_ee) \
 50 |         #     .map(clip_image)
 51 |         # Check if we have any data
 52 |         count = dw.size().getInfo()
 53 |         if count == 0:
 54 |             print("No Dynamic World data available for the specified area and date range")
 55 |         else:
 56 |             # Get median image and select forest class (class 1)
 57 |             dw_median = dw.median().clip(aoi)
 58 |             # non 1 value (==0, or >=2 ) is non forest class
 59 |             non_forest_mask = dw_median.select('label').eq(0).Or(dw_median.select('label').gte(2))
 60 |             dw_mask = ee.Image(1).clip(aoi).where(non_forest_mask, 0)
 61 |     
 62 |     # Create Forest/Non-Forest mask if requested
 63 |     if mask_type in ['FNF', 'ALL']:
 64 |         # This FNF data sets is only available until 2018
 65 |         # FNF4 is only available from 2018 to 2021
 66 |         # mannually assign FNF start and end date to 2018-01-01 and 2021-12-31
 67 |         fnf_start_date = ee.Date('2020-01-01')
 68 |         fnf_end_date = ee.Date('2020-12-31')
 69 |         fnf = ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/FNF4") \
 70 |             .filterBounds(buffered_aoi) \
 71 |             .filterDate(fnf_start_date, fnf_end_date)
 72 |         
 73 |         # # Import ALOS/PALSAR dataset using buffered AOI
 74 |         # fnf = ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/FNF4") \
 75 |         #     .filterBounds(buffered_aoi) \
 76 |         #     .filterDate(start_date_ee, end_date_ee)
 77 |         # fnf = ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/FNF") \
 78 |         #     .filterDate(start_date_ee, end_date_ee) \
 79 |         #     .map(clip_image)
 80 |         
 81 |         # Check if we have any data
 82 |         count = fnf.size().getInfo()
 83 |         if count == 0:
 84 |             print("No Dense ALOS/PALSAR FNF4 data available for the specified area and date range")
 85 |             fnf_start_date = ee.Date('2017-01-01')
 86 |             fnf_end_date = ee.Date('2017-12-31')
 87 |             fnf = ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/FNF") \
 88 |                 .filterBounds(buffered_aoi) \
 89 |                 .filterDate(start_date_ee, end_date_ee)
 90 |             count = fnf.size().getInfo()
 91 |             if count == 0:
 92 |                 print("No ALOS/PALSAR FNF data available for the specified area and date range")
 93 |             else:
 94 |                 print("ALOS/PALSAR FNF data available for the specified area and date range")
 95 |                 fnf_median = fnf.median().clip(aoi)
 96 |                 fnf_forest = fnf_median.select('fnf').eq(1)
 97 |         else:
 98 |             print("Dense ALOS/PALSAR FNF4 data available for the specified area and date range")
 99 |             # Get median image and process forest mask
100 |             fnf_median = fnf.median().clip(aoi)
101 |             fnf_forest = fnf_median.select('fnf').eq(1).Or(fnf_median.select('fnf').eq(2))
102 |         
103 |         fnf_mask = ee.Image(0).clip(aoi).where(fnf_forest, 1)
104 |     
105 |     if mask_type in ['WC', 'ALL']:
106 |         # 2021-01-01T00:00:00Z–2022-01-01T00:00:00Z
107 |         wc = ee.ImageCollection("ESA/WorldCover/v200").first() 
108 |             # .filterBounds(buffered_aoi) \
109 |             # .filterDate(ee.Date('2021-01-01'), ee.Date('2022-01-01'))
110 |         wc = wc.clip(aoi)
111 |         wc_tree = wc.eq(10)
112 |         wc_mask = ee.Image(0).clip(aoi).where(wc_tree, 1)
113 |         
114 |     # Create NDVI-based mask if requested
115 |     if mask_type in ['NDVI', 'ALL']:
116 |         # Import Sentinel-2 dataset using buffered AOI
117 |         s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
118 |             .filterBounds(buffered_aoi) \
119 |             .filterDate(start_date_ee, end_date_ee)
120 |         # Get cloud probability data
121 |         S2_CLOUD_PROBABILITY = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
122 |             .filterDate(start_date_ee, end_date_ee) \
123 |             .filterBounds(aoi)
124 |             
125 |         s2 = ee.ImageCollection(s2) \
126 |                     .map(lambda img: img.addBands(S2_CLOUD_PROBABILITY.filter(ee.Filter.equals('system:index', img.get('system:index'))).first()))
127 |         def maskClouds(img):
128 |             clouds = ee.Image(img).select('probability')
129 |             # ee.Image(img.get('cloud_mask')).select('probability')
130 |             isNotCloud = clouds.lt(70)
131 |             return img.mask(isNotCloud)
132 |         
133 |         def maskEdges(s2_img):
134 |             return s2_img.updateMask(
135 |                 s2_img.select('B8A').mask().updateMask(s2_img.select('B9').mask())
136 |             )#.updateMask(mask_raster.eq(1))
137 | 
138 |         s2 = s2.map(maskEdges) 
139 |         s2 = s2.map(maskClouds) 
140 |         
141 |         # Check if we have any data
142 |         count = s2.size().getInfo()
143 |         if count == 0:
144 |             print("No Sentinel-2 data available for the specified area and date range for NDVI calculation")
145 |         else:
146 |             # Calculate NDVI for each image
147 |             def add_ndvi(img):
148 |                 ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
149 |                 return img.addBands(ndvi)
150 |             
151 |             s2_with_ndvi = s2.map(add_ndvi)
152 |             
153 |             # Get median NDVI
154 |             ndvi_median = s2_with_ndvi.select('NDVI').median().clip(aoi)
155 |             
156 |             # Create forest mask based on NDVI threshold or Nodata
157 |             ndvi_forest = ndvi_median.gte(ndvi_threshold)
158 |             ndvi_mask = ee.Image(0).clip(aoi).where(ndvi_forest, 1)
159 |     
160 |     # if mask_type in ['CHM', 'ALL']:
161 |     canopy_ht_2023 = ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight") \
162 |         .filterBounds(aoi).mosaic().select([0],['ht2023'])
163 |     # canopy height >= 1m is tree or nodata
164 |     ch_tree = canopy_ht_2023.gte(1)
165 |     ch_mask = ee.Image(0).clip(aoi).where(ch_tree, 1)
166 |     
167 |     # Determine final mask based on mask_type
168 |     if mask_type == 'DW':
169 |         forest_mask = dw_mask
170 |     elif mask_type == 'FNF':
171 |         forest_mask = fnf_mask
172 |     elif mask_type == 'NDVI':
173 |         forest_mask = ndvi_mask
174 |     elif mask_type == 'WC':
175 |         forest_mask = wc_mask.And(ch_mask).And(ndvi_mask)
176 |     elif mask_type == 'CHM':
177 |         forest_mask = ch_mask
178 |     elif mask_type == 'ALL':
179 |         # Combine all masks (if ANY mask indicates forest, treat as forest)
180 |         forest_mask = dw_mask.Or(fnf_mask).Or(wc_mask)#.Or(ndvi_mask).
181 |         forest_mask = forest_mask.And(ch_mask).And(ndvi_mask)
182 |         # .And(fnf_mask)
183 |     else:
184 |         forest_mask = ee.Image(1).clip(aoi)
185 |     
186 |     return forest_mask
187 | 
188 | def apply_forest_mask(
189 |         # data: Union[ee.FeatureCollection, ee.Image, ee.ImageCollection, ee.Geometry, pd.DataFrame, gpd.GeoDataFrame],
190 |         data: Union[ee.FeatureCollection, ee.Image, ee.ImageCollection, ee.Geometry],
191 |         mask_type: str,
192 |         aoi: ee.Geometry,
193 |         year: int,
194 |         start_date: str,
195 |         end_date: str,
196 |         ndvi_threshold: float = 0.2,
197 |         scale: int = 30,
198 |     ) -> Union[ee.FeatureCollection, ee.Image, ee.ImageCollection]:
199 |         """
200 |         Apply forest mask to the data.
201 |         
202 |         Args:
203 |             data: Input data (FeatureCollection, Image, ImageCollection, Geometry, DataFrame, or GeoDataFrame)
204 |             mask_type: Type of mask to apply ('DW', 'FNF', 'NDVI', 'ALL', 'none')
205 |             aoi: Area of interest as Earth Engine Geometry
206 |             year: Year for analysis
207 |             start_date: Start date for Sentinel-2 data (for Earth Engine data)
208 |             end_date: End date for Sentinel-2 data (for Earth Engine data)
209 |             ndvi_threshold: NDVI threshold for forest classification (default: 0.2)
210 |         
211 |         Returns:
212 |             Masked data of the same type as input. For DataFrames/GeoDataFrames, returns a copy with
213 |             non-forest areas having rh=0. For Earth Engine data, returns masked data with masked out non-forest areas.
214 |             
215 |         Raises:
216 |             ValueError: If mask_type is not one of 'DW', 'FNF', 'NDVI', 'ALL', or 'none'
217 |             ee.ee_exception.EEException: If no data is available for the specified area and date range
218 |         """
219 |         if mask_type not in ['DW', 'FNF', 'NDVI', 'ALL', 'none']:
220 |             raise ValueError(f"Invalid mask_type: {mask_type}. Must be one of 'DW', 'FNF', 'NDVI', 'ALL', or 'none'")
221 |         
222 |         # Format dates properly for Earth Engine
223 |         start_date_ee = ee.Date(f'{year}-{start_date}')
224 |         end_date_ee = ee.Date(f'{year}-{end_date}')
225 |         
226 |         # Create forest mask
227 |         forest_mask = create_forest_mask(mask_type, aoi, start_date_ee, end_date_ee, ndvi_threshold)
228 |         
229 |         # Filter features that intersect with the forest mask
230 |         binary_forest_mask = forest_mask.gt(0.0)
231 |         
232 |         def update_forest_mask(feature_or_image):
233 |             """Update forest mask for a feature or image using server-side operations."""
234 |             element = ee.Element(feature_or_image)
235 |             element_type = ee.Algorithms.ObjectType(element)
236 |             
237 |             def handle_feature():
238 |                 # Get the feature and preserve all properties
239 |                 feature = ee.Feature(element)
240 |                 props = feature.toDictionary()
241 |                 
242 |                 # Check if point is in forest
243 |                 is_forest = binary_forest_mask.reduceRegion(
244 |                     reducer=ee.Reducer.first(),
245 |                     geometry=feature.geometry(),
246 |                     scale=scale
247 |                 ).get(binary_forest_mask.bandNames().get(0))
248 |                 
249 |                 # Update height based on forest mask
250 |                 height = ee.Algorithms.If(
251 |                     ee.Algorithms.IsEqual(is_forest, 1),
252 |                     feature.get('rh'),
253 |                     0
254 |                 )
255 |                 
256 |                 # Keep the original properties but update rh
257 |                 return ee.Feature(feature.geometry(), props.set('rh', height))
258 |             
259 |             def handle_image():
260 |                 return ee.Image(element).updateMask(binary_forest_mask)
261 |             
262 |             return ee.Algorithms.If(
263 |                 ee.Algorithms.IsEqual(element_type, 'Feature'),
264 |                 handle_feature(),
265 |                 handle_image()
266 |             )
267 |         
268 |         # Apply the mask based on data type
269 |         if isinstance(data, ee.FeatureCollection):
270 |             masked_data = data.map(update_forest_mask)
271 |         elif isinstance(data, ee.ImageCollection):
272 |             masked_data = data.map(update_forest_mask)
273 |         elif isinstance(data, (ee.Image, ee.Geometry)):
274 |             masked_data = update_forest_mask(data)
275 |         else:
276 |             # # Handle pandas DataFrame or GeoDataFrame
277 |             # if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
278 |             #     if 'rh' not in data.columns:
279 |             #         raise ValueError("DataFrame must contain 'rh' column")
280 |                 
281 |             #     # Create a binary mask for forest (all True for 'none' mask type)
282 |             #     if mask_type == 'none':
283 |             #         forest_mask = pd.Series(True, index=data.index)
284 |             #     else:
285 |             #         # Apply masking criteria based on mask_type
286 |             #         forest_mask = pd.Series(True, index=data.index)
287 |                     
288 |             #         if 'ndvi' in data.columns and mask_type in ['NDVI', 'ALL']:
289 |             #             forest_mask &= data['ndvi'] >= ndvi_threshold
290 |                     
291 |             #         if 'dw_class' in data.columns and mask_type in ['DW', 'ALL']:
292 |             #             forest_mask &= data['dw_class'] == 1
293 |                     
294 |             #         if 'fnf' in data.columns and mask_type in ['FNF', 'ALL']:
295 |             #             forest_mask &= data['fnf'].isin([1, 2])  # Assuming 1,2 are forest classes
296 |                 
297 |             #     # Apply mask and set non-forest heights to 0
298 |             #     masked_data = data.copy()
299 |             #     masked_data.loc[~forest_mask, 'rh'] = 0
300 |             #     return masked_data
301 |             # else:
302 |             print(f"Unsupported data type: {type(data)}")
303 |             raise ValueError(f"Invalid data type: {type(data)}. Must be one of ee.FeatureCollection, ee.Image, ee.ImageCollection, ee.Geometry, pd.DataFrame, or gpd.GeoDataFrame")
304 |         
305 |         return masked_data
306 | 


--------------------------------------------------------------------------------
/for_upload_download.py:
--------------------------------------------------------------------------------
  1 | import ee
  2 | import geemap
  3 | import os
  4 | from typing import Union, List, Dict
  5 | from datetime import datetime
  6 | 
  7 | def export_to_asset(
  8 |     image: ee.Image,
  9 |     description: str,
 10 |     asset_id: str,
 11 |     scale: int = 30,
 12 |     max_pixels: int = 1e13
 13 | ) -> Dict[str, str]:
 14 |     """
 15 |     Export an Earth Engine image to an Earth Engine asset.
 16 |     
 17 |     Args:
 18 |         image: Earth Engine image to export
 19 |         description: Description of the export task
 20 |         asset_id: Asset ID for the export
 21 |         scale: Scale in meters
 22 |         max_pixels: Maximum number of pixels to export
 23 |     
 24 |     Returns:
 25 |         Dict[str, str]: Task information
 26 |     """
 27 |     task = ee.batch.Export.image.toAsset(
 28 |         image=image,
 29 |         description=description,
 30 |         assetId=asset_id,
 31 |         scale=scale,
 32 |         maxPixels=max_pixels
 33 |     )
 34 |     
 35 |     task.start()
 36 |     
 37 |     return {
 38 |         'task_id': task.id,
 39 |         'description': description,
 40 |         'asset_id': asset_id
 41 |     }
 42 | 
 43 | def export_to_drive(
 44 |     image: ee.Image,
 45 |     description: str,
 46 |     folder: str,
 47 |     file_name: str,
 48 |     scale: int = 30,
 49 |     max_pixels: int = 1e13,
 50 |     file_format: str = 'GeoTIFF'
 51 | ) -> Dict[str, str]:
 52 |     """
 53 |     Export an Earth Engine image to Google Drive.
 54 |     
 55 |     Args:
 56 |         image: Earth Engine image to export
 57 |         description: Description of the export task
 58 |         folder: Google Drive folder name
 59 |         file_name: Output file name
 60 |         scale: Scale in meters
 61 |         max_pixels: Maximum number of pixels to export
 62 |         file_format: Output file format
 63 |     
 64 |     Returns:
 65 |         Dict[str, str]: Task information
 66 |     """
 67 |     task = ee.batch.Export.image.toDrive(
 68 |         image=image,
 69 |         description=description,
 70 |         folder=folder,
 71 |         fileNamePrefix=file_name,
 72 |         scale=scale,
 73 |         maxPixels=max_pixels,
 74 |         fileFormat=file_format
 75 |     )
 76 |     
 77 |     task.start()
 78 |     
 79 |     return {
 80 |         'task_id': task.id,
 81 |         'description': description,
 82 |         'folder': folder,
 83 |         'file_name': file_name
 84 |     }
 85 | 
 86 | def download_to_local(
 87 |     image: ee.Image,
 88 |     output_dir: str,
 89 |     file_name: str,
 90 |     scale: int = 30,
 91 |     region: ee.Geometry = None
 92 | ) -> str:
 93 |     """
 94 |     Download an Earth Engine image to local storage.
 95 |     
 96 |     Args:
 97 |         image: Earth Engine image to download
 98 |         output_dir: Local output directory
 99 |         file_name: Output file name
100 |         scale: Scale in meters
101 |         region: Region to download (if None, uses image bounds)
102 |     
103 |     Returns:
104 |         str: Path to downloaded file
105 |     """
106 |     # Create output directory if it doesn't exist
107 |     os.makedirs(output_dir, exist_ok=True)
108 |     
109 |     # Get download URL
110 |     if region is None:
111 |         region = image.geometry()
112 |     
113 |     url = image.getDownloadURL({
114 |         'scale': scale,
115 |         'region': region,
116 |         'format': 'GeoTIFF'
117 |     })
118 |     
119 |     # Download file
120 |     output_path = os.path.join(output_dir, f"{file_name}.tif")
121 |     geemap.download_ee_image(url, output_path)
122 |     
123 |     return output_path 


--------------------------------------------------------------------------------
/l2a_gedi_source.py:
--------------------------------------------------------------------------------
  1 | import ee
  2 | from typing import Union, List
  3 | 
  4 | def get_gedi_data(
  5 |     aoi: ee.Geometry,
  6 |     start_date: str,
  7 |     end_date: str,
  8 |     quantile: str
  9 | ) -> ee.ImageCollection:
 10 |     """
 11 |     Get GEDI L2A data for the specified area and time period.
 12 |     
 13 |     Args:
 14 |         aoi: Area of interest as Earth Engine Geometry
 15 |         start_date: Start date for GEDI data
 16 |         end_date: End date for GEDI data
 17 |         quantile: Quantile for GEDI data (e.g., 'rh100')
 18 |     
 19 |     Returns:
 20 |         ee.ImageCollection: GEDI data points
 21 |     """
 22 |     # Import GEDI L2A dataset
 23 |     gedi = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')
 24 |     
 25 |     # Filter by date and region
 26 |     gedi_filtered = gedi.filterDate(start_date, end_date) \
 27 |                         .filterBounds(aoi)
 28 | 
 29 |     # Select quality metrics and height metrics
 30 |     # gedi_filtered = gedi_filtered.select([
 31 |     #     'quality_flag',
 32 |     #     'degrade_flag',
 33 |     #     'sensitivity',
 34 |     #     'solar_elevation',
 35 |     #     'rh100',
 36 |     #     'rh98',
 37 |     #     'rh95',
 38 |     #     'rh90',
 39 |     #     'rh75',
 40 |     #     'rh50',
 41 |     #     'rh25',
 42 |     #     'rh10'
 43 |     # ])
 44 |     
 45 |     # https://lpdaac.usgs.gov/documents/982/gedi_l2a_dictionary_P003_v2.html
 46 |     def qualityMask(img):
 47 |         # First check if we have valid data
 48 |         has_data = img.select(quantile).gt(0)
 49 |         # Then apply quality filters
 50 |         quality_ok = img.select("quality_flag").eq(1) # Only select good quality data
 51 |         degrade_ok = img.select("degrade_flag").eq(0) # Only select no degrade data
 52 |         sensitivity_ok = img.select('sensitivity').gt(0.95) # Only select high sensitivity data 0.90 is the minimum
 53 |         fullPowerBeam_ok = img.select('beam').gt(4) # Only select full power beam with BEAM0101, BEAM0110, BEAM1000, BEAM1011
 54 |         solar_elevation_ok = img.select('solar_elevation').lte(0) # less tan 0 lead to day time distorsion by sun light
 55 |         detect_node_ok = img.select('num_detectedmodes').gt(0) # Only select detect node data
 56 |         # Only select data with elevation difference less than 50m
 57 |         elev_diff_ok = img.select('elev_lowestmode').subtract(img.select('digital_elevation_model_srtm')).lt(50) \
 58 |             # .And(img.select('elev_lowestmode').subtract(img.select('digital_elevation_model_srtm')).lt(-50))
 59 |         
 60 |         # ‘local_beam_elevatio’ < 1.5◦
 61 |         # local_beam_elevation_ok = img.select('local_beam_elevation').lt(1.5)
 62 | 
 63 |         # Combine all conditions
 64 |         return img.updateMask(has_data) \
 65 |                  .updateMask(quality_ok) \
 66 |                  .updateMask(degrade_ok) \
 67 |                 .updateMask(sensitivity_ok) \
 68 |                 .updateMask(fullPowerBeam_ok) \
 69 |                 .updateMask(solar_elevation_ok) \
 70 |                 .updateMask(detect_node_ok) \
 71 |                 .updateMask(elev_diff_ok) \
 72 |                     
 73 |                 # .updateMask(local_beam_elevation_ok) \
 74 |                     
 75 |     
 76 |     # Select and rename the quantile
 77 |     # def rename_property(image):
 78 |     #     return image.select([quantile]).rename('rh')
 79 |     
 80 |     # gedi_filtered = gedi_filtered.map(rename_property)
 81 |     
 82 |     # Then apply quality mask
 83 |     gedi_filtered = gedi_filtered.map(qualityMask)
 84 |     gedi_filtered_rh = gedi_filtered.select(quantile).mosaic().rename("rh")
 85 |     gedi_additional_bands = [
 86 |         'digital_elevation_model',
 87 |         'digital_elevation_model_srtm',
 88 |         'elev_lowestmode',
 89 |     ]
 90 |     gedi_filtered = gedi_filtered.select(gedi_additional_bands).mosaic().addBands(gedi_filtered_rh)
 91 |     # Get all valid points by using reduce(ee.Reducer.toCollection())
 92 |     # Specify the property names we want to keep
 93 |     # gedi_points = gedi_filtered.select([quantile, 'quality_flag', 'degrade_flag']).reduce(
 94 |     #     ee.Reducer.toCollection([quantile])
 95 |     #     # ee.Reducer.toCollection(['quality_flag', 'degrade_flag', quantile])
 96 |     # )
 97 |     
 98 |     # Rename the quantile band to 'rh'
 99 |     # gedi_points = gedi_points.rename(quantile, "rh")
100 |     
101 |     return gedi_filtered


--------------------------------------------------------------------------------
/raster_utils.py:
--------------------------------------------------------------------------------
  1 | """Shared raster processing utilities."""
  2 | 
  3 | import rasterio
  4 | from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
  5 | from rasterio.windows import Window, from_bounds
  6 | import numpy as np
  7 | import os
  8 | 
  9 | def clip_and_resample_raster(src_path: str, bounds: tuple, target_transform=None, 
 10 |                            target_crs=None, target_shape=None, output_path: str = None):
 11 |     """Clip raster to bounds and optionally resample to target resolution."""
 12 |     with rasterio.open(src_path) as src:
 13 |         # Transform bounds if CRS differs
 14 |         if target_crs and src.crs != target_crs:
 15 |             clip_bounds = transform_bounds(target_crs, src.crs, *bounds)
 16 |         else:
 17 |             clip_bounds = bounds
 18 |         
 19 |         # Calculate output dimensions based on bounds and resolution
 20 |         west, south, east, north = clip_bounds
 21 |         output_width = max(1, int(round((east - west) / abs(src.res[0]))))
 22 |         output_height = max(1, int(round((north - south) / abs(src.res[1]))))
 23 |         
 24 |         # Create window for clipping
 25 |         col_start = int((west - src.bounds.left) / src.res[0])
 26 |         row_start = int((src.bounds.top - north) / src.res[1])
 27 |         col_stop = int((east - src.bounds.left) / src.res[0])
 28 |         row_stop = int((src.bounds.top - south) / src.res[1])
 29 |         
 30 |         window = Window(
 31 |             col_off=col_start,
 32 |             row_off=row_start,
 33 |             width=max(1, col_stop - col_start),
 34 |             height=max(1, row_stop - row_start)
 35 |         )
 36 |         
 37 |         # Read data in window
 38 |         data = src.read(1, window=window)
 39 |         
 40 |         # Get transform for clipped data
 41 |         clip_transform = rasterio.transform.from_bounds(
 42 |             west, south, east, north,
 43 |             output_width, output_height
 44 |         )
 45 |         
 46 |         # If target parameters are provided, resample the data
 47 |         if all(x is not None for x in [target_transform, target_crs, target_shape]):
 48 |             # Create destination array
 49 |             dest = np.zeros(target_shape, dtype=np.float32)
 50 |             
 51 |             # Reproject and resample
 52 |             reproject(
 53 |                 source=data,
 54 |                 destination=dest,
 55 |                 src_transform=clip_transform,
 56 |                 src_crs=src.crs,
 57 |                 dst_transform=target_transform,
 58 |                 dst_crs=target_crs,
 59 |                 resampling=Resampling.average
 60 |             )
 61 |             
 62 |             data = dest
 63 |             out_transform = target_transform
 64 |             out_crs = target_crs
 65 |         else:
 66 |             out_transform = clip_transform
 67 |             out_crs = src.crs
 68 |         
 69 |         # Save if output path provided
 70 |         if output_path:
 71 |             profile = src.profile.copy()
 72 |             profile.update({
 73 |                 'height': data.shape[0],
 74 |                 'width': data.shape[1],
 75 |                 'transform': out_transform,
 76 |                 'crs': out_crs
 77 |             })
 78 |             
 79 |             with rasterio.open(output_path, 'w', **profile) as dst:
 80 |                 dst.write(data, 1)
 81 |         
 82 |         return data, out_transform
 83 | def get_common_bounds(pred_path: str, ref_path: str):
 84 |     """Get intersection bounds of two rasters in the prediction CRS."""
 85 |     with rasterio.open(pred_path) as pred_src:
 86 |         pred_crs = pred_src.crs
 87 |         pred_bounds = pred_src.bounds
 88 |         print(f"\nPrediction bounds ({pred_crs}):")
 89 |         print(f"Left: {pred_bounds.left:.6f}, Bottom: {pred_bounds.bottom:.6f}")
 90 |         print(f"Right: {pred_bounds.right:.6f}, Top: {pred_bounds.top:.6f}")
 91 |         
 92 |         with rasterio.open(ref_path) as ref_src:
 93 |             print(f"\nReference bounds ({ref_src.crs}):")
 94 |             print(f"Left: {ref_src.bounds.left:.6f}, Bottom: {ref_src.bounds.bottom:.6f}")
 95 |             print(f"Right: {ref_src.bounds.right:.6f}, Top: {ref_src.bounds.top:.6f}")
 96 |             
 97 |             if ref_src.crs != pred_crs:
 98 |                 print(f"\nTransforming reference bounds to {pred_crs}")
 99 |                 ref_bounds = transform_bounds(ref_src.crs, pred_crs, *ref_src.bounds)
100 |             else:
101 |                 ref_bounds = ref_src.bounds
102 |             
103 |             # Find intersection
104 |             west = max(pred_bounds.left, ref_bounds[0])
105 |             south = max(pred_bounds.bottom, ref_bounds[1])
106 |             east = min(pred_bounds.right, ref_bounds[2])
107 |             north = min(pred_bounds.top, ref_bounds[3])
108 |             
109 |             bounds = (west, south, east, north)
110 |             print(f"\nIntersection bounds: {bounds}")
111 |             return bounds
112 | 
113 | 
114 | def load_and_align_rasters(pred_path: str, ref_path: str, forest_mask_path: str = None, output_dir: str = None):
115 |     """Load and align rasters to same CRS and resolution, optionally applying forest mask."""
116 |     # Get intersection bounds in prediction CRS
117 |     bounds = get_common_bounds(pred_path, ref_path)
118 |     
119 |     # Get prediction properties to use as target
120 |     with rasterio.open(pred_path) as pred_src:
121 |         target_transform = pred_src.transform
122 |         target_crs = pred_src.crs
123 |         target_shape = pred_src.shape
124 |     
125 |     pred_filename = os.path.basename(pred_path)
126 |     ref_filename = os.path.basename(ref_path)
127 |     
128 |  
129 |     if output_dir:
130 |         # Create paths for clipped files
131 |         pred_clip_path = os.path.join(output_dir, f"{os.path.splitext(pred_filename)[0]}_clipped.tif")
132 |         ref_clip_path = os.path.join(output_dir, f"{os.path.splitext(ref_filename)[0]}_clipped.tif")
133 |     else:
134 |         pred_clip_path = ref_clip_path = None
135 |         
136 |     print("\nProcessing prediction raster...")
137 |     pred_data, _ = clip_and_resample_raster(
138 |         pred_path, bounds,
139 |         target_transform=target_transform,
140 |         target_crs=target_crs,
141 |         target_shape=target_shape,
142 |         output_path=pred_clip_path
143 |     )
144 |     
145 |     print("\nProcessing reference raster...")
146 |     ref_data, _ = clip_and_resample_raster(
147 |         ref_path, bounds,
148 |         target_transform=target_transform,
149 |         target_crs=target_crs,
150 |         target_shape=target_shape,
151 |         output_path=ref_clip_path
152 |     )
153 |     
154 |     # Load and apply forest mask if provided
155 |     forest_mask = None
156 |     if forest_mask_path and os.path.exists(forest_mask_path):
157 |         print("\nProcessing forest mask...")
158 |         mask_data, _ = clip_and_resample_raster(
159 |             forest_mask_path, bounds,
160 |             target_transform=target_transform,
161 |             target_crs=target_crs,
162 |             target_shape=target_shape
163 |         )
164 |         # Create binary mask
165 |         forest_mask = (mask_data > 0)
166 |         
167 |         # Apply forest mask to both prediction and reference data
168 |         pred_data[~forest_mask] = np.nan
169 |         ref_data[~forest_mask] = np.nan
170 |         
171 |         print(f"Forest mask applied - {np.sum(forest_mask):,} forest pixels")
172 | 
173 |     return pred_data, ref_data, target_transform, forest_mask


--------------------------------------------------------------------------------
/requirements.txt:
--------------------------------------------------------------------------------
 1 | earthengine-api>=0.1.323
 2 | geemap>=0.19.0
 3 | numpy>=1.21.0
 4 | pandas>=1.3.0
 5 | matplotlib>=3.4.0
 6 | jupyter>=1.0.0
 7 | notebook>=6.4.0
 8 | ipywidgets>=7.6.0 
 9 | mss
10 | pytest
11 | rasterio>=1.3.0
12 | tqdm>=4.65.0
13 | scikit-learn
14 | seaborn
15 | shapely
16 | reportlab
17 | geopandas


--------------------------------------------------------------------------------
/run_main.py:
--------------------------------------------------------------------------------
  1 | import os
  2 | import subprocess
  3 | from pathlib import Path
  4 | import time
  5 | 
  6 | from utils import get_latest_file
  7 | from combine_heights import combine_heights_with_training
  8 | 
  9 | # Set parameters
 10 | def main(type: str):
 11 | 
 12 |     aoi_path = 'downloads/new_aoi.geojson'
 13 |     if not os.path.exists(aoi_path):
 14 |         raise FileNotFoundError(f"AOI file not found at {aoi_path}")
 15 | 
 16 |     # Create output directories
 17 |     output_dir = 'chm_outputs'
 18 |     eval_dir = os.path.join(output_dir, 'evaluation')
 19 |     os.makedirs(output_dir, exist_ok=True)
 20 |     os.makedirs(eval_dir, exist_ok=True)
 21 | 
 22 |     # Build command for GEE model training and prediction
 23 |     gee_cmd = [
 24 |         'python', 'chm_main.py',
 25 |         '--aoi', aoi_path,
 26 |         '--year', '2022',
 27 |         '--start-date', '01-01',
 28 |         '--end-date', '12-31',
 29 |         '--gedi-start-date', '2022-06-01', 
 30 |         '--gedi-end-date', '2022-08-31',
 31 |         '--buffer', '10000',
 32 |         '--clouds-th', '70',
 33 |         '--quantile', 'rh98',
 34 |         '--model', 'RF',
 35 |         '--num-trees-rf', '500',
 36 |         '--min-leaf-pop-rf', '5',
 37 |         '--bag-frac-rf', '0.5',
 38 |         '--max-nodes-rf', '1000',
 39 |         '--output-dir', output_dir,
 40 |         # '--export-training',
 41 |         # '--export-predictions',
 42 |         # '--export-stack',
 43 |         '--export-forest-mask',
 44 |         '--scale', '10',
 45 |         # '--resample', 'bicubic',
 46 |         '--ndvi-threshold', '0.35',
 47 |         # '--mask-type', 'ALL',
 48 |         '--mask-type', 'WC',
 49 |     ]
 50 |     if type == 'data_preparation':
 51 |         # Convert all arguments to strings
 52 |         gee_cmd = [str(arg) for arg in gee_cmd]
 53 |         # Run the GEE model training and prediction
 54 |         print("Running GEE canopy height model...")
 55 |         subprocess.run(gee_cmd, check=True)
 56 |         # # Wait for downloaded files
 57 |         # print("Waiting for GEE exports to complete...")
 58 |         # time.sleep(60)  # Wait for files to be downloaded
 59 |     else:
 60 |         # Get the most recent training data, stack, and mask files
 61 |         training_file = get_latest_file(output_dir, 'training_data')
 62 |         stack_file = get_latest_file(output_dir, 'stack')
 63 |         mask_file = get_latest_file(output_dir, 'forestMask')
 64 |         buffered_mask_file = get_latest_file(output_dir, 'buffered_forestMask')
 65 |         # Forest mask is optional
 66 | 
 67 |     ref_file = os.path.join('downloads', 'dchm_09id4.tif')
 68 |     
 69 |     if type == 'height_analysis':
 70 |         combine_heights_with_training(output_dir, ref_file)
 71 | 
 72 |     if type =='train_predict':
 73 |         try:
 74 |             # Build command for local model training and prediction
 75 |             train_cmd = [
 76 |                 'python', 'train_predict_map.py',
 77 |                 '--training-data', training_file,
 78 |                 '--stack', stack_file,
 79 |                 '--mask', mask_file,  # Used as both quality mask and forest mask
 80 |                 '--buffered-mask', buffered_mask_file,
 81 |                 '--output-dir', output_dir,
 82 |                 # '--output-filename', 'local_canopy_height_predictions.tif',
 83 |                 '--test-size', '0.1',
 84 |                 '--apply-forest-mask',  # Add flag to indicate mask should be used as forest mask
 85 |                 # '--model', 'mlp', # default is 'rf'
 86 |                 # '--batch_size', '32', # default is 64
 87 |                 '--ch_col', 'rh',
 88 |             ]
 89 |             # Run local training and prediction
 90 |             print("\nRunning local model training and prediction...")
 91 |             subprocess.run([str(arg) for arg in train_cmd], check=True)
 92 |         except FileNotFoundError as e:
 93 |             print(f"Error: {e}")
 94 |             print("Please ensure all required files have been exported from GEE before running local processing.")
 95 |     
 96 |     if type == 'evaluate':
 97 |         pred_file = get_latest_file(output_dir, 'predictCH') 
 98 |         # Run evaluation with PDF report generation
 99 |         eval_cmd = [
100 |             'python', 'evaluate_predictions.py',
101 |             '--pred', pred_file,
102 |             '--ref', ref_file,
103 |             '--output', eval_dir,
104 |             '--pdf',
105 |             '--training', training_file,
106 |             '--merged', stack_file,
107 |             '--forest-mask', mask_file
108 |         ]
109 |         print("\nRunning evaluation...")
110 |         subprocess.run([str(arg) for arg in eval_cmd], check=True)
111 |         print("All processing complete!")
112 | 
113 | if __name__ == "__main__":
114 |     # Example usage
115 |     # main('data_preparation')
116 |     main('height_analysis')
117 |     # main('train_predict')
118 |     # main('evaluate')
119 |     


--------------------------------------------------------------------------------
/save_evaluation_pdf.py:
--------------------------------------------------------------------------------
  1 | """Module for generating PDF evaluation reports."""
  2 | 
  3 | import os
  4 | import json
  5 | import pandas as pd
  6 | import numpy as np
  7 | import rasterio
  8 | from datetime import datetime
  9 | from reportlab.pdfgen import canvas
 10 | from reportlab.lib.pagesizes import letter
 11 | from reportlab.lib import colors
 12 | from reportlab.graphics.shapes import Drawing
 13 | from reportlab.graphics.charts.barcharts import VerticalBarChart
 14 | from reportlab.graphics.charts.legends import Legend
 15 | from rasterio.crs import CRS
 16 | from rasterio.warp import transform_bounds
 17 | 
 18 | from raster_utils import load_and_align_rasters
 19 | from utils import get_latest_file
 20 | 
 21 | 
 22 | def scale_adjust_band(band_data, min_val, max_val, contrast=1.0, gamma=1.0):
 23 |     """Adjust band data with min/max scaling, contrast, and gamma."""
 24 |     # Handle NaN values
 25 |     nan_mask = np.isnan(band_data)
 26 |     temp_nodata = -9999
 27 |     work_data = band_data.copy()
 28 |     
 29 |     if np.any(work_data[~nan_mask] == temp_nodata):
 30 |         valid_min = np.min(work_data[~nan_mask]) if not nan_mask.all() else -1
 31 |         temp_nodata = valid_min - 1
 32 | 
 33 |     work_data[nan_mask] = temp_nodata
 34 |     work_data = work_data.astype(np.float32)
 35 | 
 36 |     # Min/Max scaling
 37 |     mask_valid = (work_data != temp_nodata)
 38 |     scaled_data = np.zeros_like(work_data, dtype=np.float32)
 39 |     if max_val - min_val != 0:
 40 |         scaled_data[mask_valid] = (work_data[mask_valid] - min_val) / (max_val - min_val)
 41 |     scaled_data[mask_valid] = np.clip(scaled_data[mask_valid], 0, 1)
 42 | 
 43 |     # Contrast adjustment
 44 |     if contrast != 1.0:
 45 |         scaled_data[mask_valid] = 0.5 + contrast * (scaled_data[mask_valid] - 0.5)
 46 |         scaled_data[mask_valid] = np.clip(scaled_data[mask_valid], 0, 1)
 47 | 
 48 |     # Gamma correction
 49 |     if gamma != 1.0 and gamma > 0:
 50 |         gamma_mask = mask_valid & (scaled_data > 0)
 51 |         with np.errstate(invalid='ignore'):
 52 |             scaled_data[gamma_mask] = scaled_data[gamma_mask]**(1.0 / gamma)
 53 |         scaled_data[gamma_mask] = np.clip(scaled_data[gamma_mask], 0, 1)
 54 | 
 55 |     # Convert to uint8
 56 |     scaled_data[~mask_valid] = 0
 57 |     scaled_uint8 = (scaled_data * 255).astype(np.uint8)
 58 |     return scaled_uint8
 59 | 
 60 | 
 61 | def load_rgb_composite(merged_path, target_shape, transform, temp_dir=None):
 62 |     """Load and process RGB composite from merged data."""
 63 |     merged_file_name = os.path.basename(merged_path)
 64 |     if temp_dir is None:
 65 |         temp_dir = os.path.dirname(merged_path)
 66 |     merged_clipped_path = os.path.join(temp_dir, f"{merged_file_name.split('.')[0]}_clipped.tif")
 67 |     os.makedirs(os.path.dirname(merged_clipped_path), exist_ok=True)
 68 |         
 69 |     try:
 70 |         with rasterio.open(merged_path) as src:
 71 |             if src.count >= 4:  # Check if we have enough bands
 72 |                 # Use S2 bands 4,3,2 (R,G,B) for natural color
 73 |                 rgb_bands = [3, 2, 1]  # B4 (R, 665nm), B3 (G, 560nm), B2 (B, 490nm)
 74 |                 rgb = np.zeros((target_shape[0], target_shape[1], 3), dtype=np.float32)
 75 |                 
 76 |                 from rasterio.warp import reproject, Resampling
 77 |                 for i, band in enumerate(rgb_bands):
 78 |                     band_data = src.read(band)  # Band numbers are 1-based
 79 |                     band_resampled = np.zeros(target_shape, dtype=np.float32)
 80 |                     reproject(
 81 |                         band_data,
 82 |                         band_resampled,
 83 |                         src_transform=src.transform,
 84 |                         src_crs=src.crs,
 85 |                         dst_transform=transform,
 86 |                         dst_crs=src.crs,
 87 |                         resampling=Resampling.bilinear
 88 |                     )
 89 |                     rgb[:, :, i] = band_resampled
 90 |                 
 91 |                 # Apply band-specific scaling for Sentinel-2 reflectance values
 92 |                 rgb_norm = np.zeros_like(rgb, dtype=np.uint8)
 93 |                 # Sentinel-2 L2A typical reflectance ranges
 94 |                 scale_params = [
 95 |                     {'min': 0, 'max': 3000, 'contrast': 1.2, 'gamma': 0.8},  # Red (B4)
 96 |                     {'min': 0, 'max': 3000, 'contrast': 1.2, 'gamma': 0.8},  # Green (B3)
 97 |                     {'min': 0, 'max': 3000, 'contrast': 1.2, 'gamma': 0.8}    # Blue (B2)
 98 |                 ]
 99 |                 for i in range(3):
100 |                     rgb_norm[:,:,i] = scale_adjust_band(
101 |                         rgb[:,:,i],
102 |                         scale_params[i]['min'],
103 |                         scale_params[i]['max'],
104 |                         contrast=scale_params[i]['contrast'],
105 |                         gamma=scale_params[i]['gamma']
106 |                     )
107 |                 
108 |                 # save the RGB composite
109 |                 profile = src.profile.copy()
110 |                 profile.update({
111 |                     'height': target_shape[0],
112 |                     'width': target_shape[1],
113 |                     'transform': transform,
114 |                     'count': 3,
115 |                     'dtype': 'uint8'
116 |                 })
117 |                 try:
118 |                     with rasterio.open(merged_clipped_path, 'w', **profile) as dst:
119 |                         # Write bands in correct order (R,G,B)
120 |                         dst.write(rgb_norm.transpose(2, 1, 0))
121 |                 except Exception as e:
122 |                     print(f"Warning: Could not save RGB composite: {e}")
123 |                     print(f"Attempted to save to: {merged_clipped_path}")
124 |                     # Continue even if saving fails - we can still use the RGB data in memory
125 |                 return rgb_norm
126 |     except Exception as e:
127 |         print(f"Error creating RGB composite: {e}")
128 |     return None
129 | 
130 | 
131 | def create_2x2_visualization(ref_data, pred_data, diff_data, merged_path, transform, output_path, mask=None, forest_mask=None, temp_dir=None):
132 |     """Create 2x2 grid with reference, prediction, difference and RGB data."""
133 |     
134 |     # Load RGB composite if available
135 |     rgb_norm = None
136 |     if merged_path and os.path.exists(merged_path):
137 |         rgb_norm = load_rgb_composite(merged_path, pred_data.shape, transform, temp_dir)
138 |     else:
139 |         print("Merged data not found or invalid. Skipping RGB composite creation.")
140 |     # Apply mask if provided
141 |     # Combine validity mask with forest mask if provided
142 |     final_mask = mask if mask is not None else np.ones_like(pred_data, dtype=bool)
143 |     if forest_mask is not None:
144 |         final_mask = final_mask & forest_mask
145 |         
146 |     # if final_mask is not None and rgb_norm is not None:
147 |     #     # Create a 3D mask by expanding dimensions
148 |     #     mask_3d = np.repeat(final_mask[:, :, np.newaxis], 3, axis=2)
149 |     #     # Apply mask - set masked areas to 0
150 |     #     rgb_norm = np.where(mask_3d, rgb_norm, 0)
151 |         
152 |     from evaluation_utils import create_comparison_grid
153 |     create_comparison_grid(ref_data, pred_data, diff_data, rgb_norm, output_path, 
154 |                            forest_mask=final_mask)
155 |     return output_path
156 | 
157 | 
158 | def format_band_names(bands, line_length=80):
159 |     """Format band names into multiple lines."""
160 |     lines = []
161 |     current_line = []
162 |     current_length = 0
163 |     
164 |     for band in bands:
165 |         if current_length + len(band) + 2 > line_length:  # +2 for comma and space
166 |             lines.append(', '.join(current_line))
167 |             current_line = [band]
168 |             current_length = len(band)
169 |         else:
170 |             current_line.append(band)
171 |             current_length += len(band) + 2  # +2 for comma and space
172 |             
173 |     if current_line:
174 |         lines.append(', '.join(current_line))
175 |     
176 |     return '\n'.join(lines)
177 | 
178 | def get_training_info(csv_path):
179 |     """Extract information from training data."""
180 |     if not os.path.exists(csv_path):
181 |         return {'sample_size': 0, 'band_names': [], 'height_range': (0, 0)}
182 |         
183 |     df = pd.read_csv(csv_path)
184 |     bands = [col for col in df.columns if col not in ['rh', 'longitude', 'latitude']]
185 |     
186 |     return {
187 |         'sample_size': len(df),
188 |         'band_names': sorted(bands),
189 |         'height_range': (df['rh'].min(), df['rh'].max())
190 |     }
191 | 
192 | 
193 | def calculate_area(bounds: tuple, crs: CRS):
194 |     """Calculate area in hectares from bounds."""
195 |     if crs.is_geographic:
196 |         center_lat = (bounds[1] + bounds[3]) / 2
197 |         center_lon = (bounds[0] + bounds[2]) / 2
198 |         utm_zone = int((center_lon + 180) / 6) + 1
199 |         utm_epsg = 32600 + utm_zone + (0 if center_lat >= 0 else 100)
200 |         utm_crs = CRS.from_epsg(utm_epsg)
201 |         bounds = transform_bounds(crs, utm_crs, *bounds)
202 |         
203 |     width = bounds[2] - bounds[0]
204 |     height = bounds[3] - bounds[1]
205 |     area_m2 = width * height
206 |     return area_m2 / 10000
207 | 
208 | 
209 | def create_feature_importance_chart(data, width, height):
210 |     """Create a bar chart showing feature importance."""
211 |     drawing = Drawing(width, height)
212 |     
213 |     # Sort data by importance value
214 |     sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
215 |     values = list(sorted_data.values())[:10]  # Top 10 features
216 |     names = list(sorted_data.keys())[:10]
217 |     
218 |     # Create and customize the chart
219 |     chart = VerticalBarChart()
220 |     chart.x = 50
221 |     chart.y = 50
222 |     chart.height = height - 100
223 |     chart.width = width - 100
224 |     chart.data = [values]
225 |     chart.categoryAxis.categoryNames = names
226 |     chart.categoryAxis.labels.boxAnchor = 'ne'
227 |     chart.categoryAxis.labels.angle = 45
228 |     chart.categoryAxis.labels.dx = -10
229 |     chart.categoryAxis.labels.dy = -20
230 |     chart.bars[0].fillColor = colors.blue
231 |     chart.valueAxis.valueMin = 0
232 |     chart.valueAxis.valueMax = max(values) * 1.1
233 |     chart.valueAxis.valueStep = max(values) / 5
234 |     
235 |     drawing.add(chart)
236 |     return drawing
237 | 
238 | def save_evaluation_to_pdf(pred_path, ref_path, pred_data, ref_data, metrics,
239 |                           output_dir, training_data_path=None, merged_data_path=None,
240 |                           mask=None, forest_mask=None, area_ha=None, validation_info=None, plot_paths=None):
241 |     """Create PDF report with evaluation results."""
242 |     os.makedirs(output_dir, exist_ok=True)
243 |     
244 |     # Calculate difference for visualization
245 |     diff_data = pred_data - ref_data
246 |     
247 |     # Create comparison grid visualization
248 |     grid_path = os.path.join(output_dir, 'comparison_grid.png')
249 |     with rasterio.open(pred_path) as src:
250 |         transform = src.transform
251 |     # Create a temp directory for RGB composites within output_dir
252 |     rgb_temp_dir = os.path.join(output_dir, 'rgb_temp')
253 |     os.makedirs(rgb_temp_dir, exist_ok=True)
254 |     
255 |     create_2x2_visualization(
256 |         ref_data, pred_data, diff_data,
257 |         merged_data_path, transform, grid_path,
258 |         mask=mask, forest_mask=forest_mask,
259 |         temp_dir=rgb_temp_dir
260 |     )
261 |     
262 |     # Get area if not provided
263 |     if area_ha is None:
264 |         with rasterio.open(pred_path) as src:
265 |             area_ha = calculate_area(src.bounds, src.crs)
266 |     
267 |     # Get training info
268 |     train_info = get_training_info(training_data_path) if training_data_path else {
269 |         'sample_size': 0, 'band_names': [], 'height_range': (0, 0)
270 |     }
271 |     
272 |     # Create PDF
273 |     date = datetime.now().strftime("%Y%m%d")
274 |     n_bands = len(train_info['band_names']) if train_info['band_names'] else 'X'
275 |     pdf_name = f"{date}_b{n_bands}_{int(area_ha)}ha.pdf"
276 |     pdf_path = os.path.join(output_dir, pdf_name)
277 |     
278 |     # Initialize PDF
279 |     c = canvas.Canvas(pdf_path, pagesize=letter)
280 |     width, height = letter
281 |     
282 |     # First page - Summary information
283 |     c.setFont("Helvetica-Bold", 16)
284 |     c.drawString(50, height-50, "Canopy Height Model Evaluation Report")
285 |     c.setFont("Helvetica", 12)
286 |     c.drawString(50, height-70, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
287 |     
288 |     # Add validation info
289 |     y = height-100
290 |     if validation_info:
291 |         c.setFont("Helvetica-Bold", 12)
292 |         c.drawString(50, y, "Data Statistics:")
293 |         c.setFont("Helvetica", 10)
294 |         y -= 15
295 |         c.drawString(70, y, "Prediction Data:")
296 |         y -= 15
297 |         c.drawString(90, y, f"Range: {validation_info['pred_range'][0]:.2f}m to {validation_info['pred_range'][1]:.2f}m")
298 |         y -= 15
299 |         c.drawString(90, y, f"Mean: {validation_info['pred_stats']['mean']:.2f}m, Std: {validation_info['pred_stats']['std']:.2f}m")
300 |         y -= 15
301 |         c.drawString(70, y, "Reference Data:")
302 |         y -= 15
303 |         c.drawString(90, y, f"Range: {validation_info['ref_range'][0]:.2f}m to {validation_info['ref_range'][1]:.2f}m")
304 |         y -= 15
305 |         c.drawString(90, y, f"Mean: {validation_info['ref_stats']['mean']:.2f}m, Std: {validation_info['ref_stats']['std']:.2f}m")
306 |         y -= 25
307 |     
308 |     # Add training info
309 |     c.setFont("Helvetica-Bold", 12)
310 |     c.drawString(50, y, "Training Data:")
311 |     c.setFont("Helvetica", 10)
312 |     y -= 15
313 |     c.drawString(70, y, f"Sample Size: {train_info['sample_size']:,}")
314 |     y -= 15
315 |     
316 |     # Format band names into multiple lines
317 |     c.drawString(70, y, "Input Bands:")
318 |     y -= 15
319 |     formatted_bands = format_band_names(train_info['band_names'], line_length=80)
320 |     for line in formatted_bands.split('\n'):
321 |         c.drawString(90, y, line)
322 |         y -= 15
323 |     
324 |     c.drawString(70, y, f"Height Range: {train_info['height_range'][0]:.1f}m to {train_info['height_range'][1]:.1f}m")
325 |     
326 |     # Add training metrics if available
327 |     chm_outputs_dir = os.path.dirname(os.path.dirname(output_dir))  # Get chm_outputs directory
328 |     print(f"Looking for model_evaluation.json in: {chm_outputs_dir}")
329 |     model_eval_path = get_latest_file(chm_outputs_dir, 'model_evaluation', required=False)
330 |     if model_eval_path:
331 |         print(f"Found model evaluation file at: {model_eval_path}")
332 |     else:
333 |         print("No model evaluation file found")
334 |     if model_eval_path and os.path.exists(model_eval_path):
335 |         try:
336 |             with open(model_eval_path) as f:
337 |                 model_data = json.load(f)
338 |             
339 |             if 'train_metrics' in model_data:
340 |                 y -= 25
341 |                 c.setFont("Helvetica-Bold", 12)
342 |                 c.drawString(50, y, "Training Metrics:")
343 |                 c.setFont("Helvetica", 10)
344 |                 y -= 15
345 |                 for metric, value in model_data['train_metrics'].items():
346 |                     metric_name = metric.replace('_', ' ').title()
347 |                     if isinstance(value, float):
348 |                         if metric.endswith('(%)'):
349 |                             c.drawString(70, y, f"{metric_name}: {value:.1f}%")
350 |                         else:
351 |                             c.drawString(70, y, f"{metric_name}: {value:.3f}")
352 |                     else:
353 |                         c.drawString(70, y, f"{metric_name}: {value}")
354 |                     y -= 15
355 |         except Exception as e:
356 |             print(f"Warning: Could not load training metrics: {e}")
357 |     
358 |     # Add area info
359 |     y -= 25
360 |     c.setFont("Helvetica-Bold", 12)
361 |     c.drawString(50, y, "Area Information:")
362 |     c.setFont("Helvetica", 10)
363 |     y -= 15
364 |     c.drawString(70, y, f"Total Area: {area_ha:,.1f} hectares")
365 |     
366 |     # Add metrics
367 |     y -= 25
368 |     c.setFont("Helvetica-Bold", 12)
369 |     c.drawString(50, y, "Evaluation Metrics:")
370 |     c.setFont("Helvetica", 10)
371 |     y -= 15
372 |     for metric, value in metrics.items():
373 |         if metric.endswith('(%)'):
374 |             c.drawString(70, y, f"{metric}: {value:,.1f}%")
375 |         else:
376 |             c.drawString(70, y, f"{metric}: {value:,.3f}")
377 |         y -= 15
378 |     
379 |     c.showPage()
380 |     
381 |     # Second page - Comparison grid
382 |     c.setFont("Helvetica-Bold", 14)
383 |     c.drawString(50, height-40, "Canopy Height Model Comparison Grid")
384 |     
385 |     if os.path.exists(grid_path):
386 |         grid_height = height - 80
387 |         grid_width = width - 100
388 |         c.drawImage(grid_path, 50, height-grid_height-40, width=grid_width, height=grid_height, preserveAspectRatio=True)
389 |     
390 |     c.showPage()
391 |     
392 |     # Third page - Analysis plots
393 |     if plot_paths:
394 |         c.setFont("Helvetica-Bold", 14)
395 |         c.drawString(50, height-40, "Detailed Analysis")
396 |         
397 |         y = height - 60
398 |         plot_height = (height - 100) / 2
399 |         
400 |         if os.path.exists(plot_paths.get('scatter', '')):
401 |             c.drawImage(plot_paths['scatter'], 50, y-plot_height, width=width/2-60, height=plot_height, preserveAspectRatio=True)
402 |         
403 |         if os.path.exists(plot_paths.get('error_hist', '')):
404 |             c.drawImage(plot_paths['error_hist'], width/2, y-plot_height, width=width/2-60, height=plot_height, preserveAspectRatio=True)
405 |         
406 |         if os.path.exists(plot_paths.get('height_dist', '')):
407 |             y -= plot_height + 20
408 |             c.drawImage(plot_paths['height_dist'], width/4, y-plot_height, width=width/2-60, height=plot_height, preserveAspectRatio=True)
409 |         
410 |         c.showPage()
411 |     
412 |     def draw_feature_importance_table(c, data, x, y, width):
413 |         """Draw a table of feature importance values."""
414 |         # Sort features by importance
415 |         sorted_features = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
416 |         total_importance = sum(sorted_features.values())
417 |         
418 |         # Calculate column widths
419 |         feature_width = width * 0.6
420 |         value_width = width * 0.2
421 |         percent_width = width * 0.2
422 |         
423 |         # Draw table border
424 |         c.rect(x, y-14, width, 30)  # Header box
425 |         
426 |         # Table header
427 |         c.setFont("Helvetica-Bold", 10)
428 |         c.drawString(x + 5, y, "Feature")
429 |         c.drawRightString(x + feature_width + value_width - 5, y, "Importance")
430 |         c.drawRightString(x + width - 5, y, "Percent")
431 |         y -= 15
432 |         
433 |         # Draw horizontal line under header
434 |         c.setLineWidth(0.5)
435 |         c.line(x, y+2, x + width, y+2)
436 |         y -= 15
437 |         
438 |         # Draw vertical lines
439 |         c.line(x + feature_width, y+32, x + feature_width, y-14*len(sorted_features))  # After Feature
440 |         c.line(x + feature_width + value_width, y+32, x + feature_width + value_width, y-14*len(sorted_features))  # After Importance
441 |         
442 |         # Table content
443 |         row = 0
444 |         c.setFont("Helvetica", 9)
445 |         for feature, importance in sorted_features.items():
446 |             # Alternate row colors
447 |             if row % 2 == 0:
448 |                 c.setFillColorRGB(0.95, 0.95, 0.95)
449 |                 c.rect(x, y-3, width, 14, fill=1, stroke=0)
450 |             c.setFillColorRGB(0, 0, 0)
451 |             
452 |             # Draw row content with padding
453 |             c.drawString(x + 5, y, feature)
454 |             c.drawRightString(x + feature_width + value_width - 5, y, f"{importance:.4f}")
455 |             percentage = (importance / total_importance) * 100
456 |             c.drawRightString(x + width - 5, y, f"{percentage:.1f}%")
457 |             
458 |             # Draw horizontal line after each row
459 |             if row < len(sorted_features) - 1:
460 |                 c.setLineWidth(0.1)
461 |                 c.line(x, y-7, x + width, y-7)
462 |             
463 |             y -= 14
464 |             row += 1
465 |             
466 |             if y < 50:  # Start new column if near bottom
467 |                 y = height - 100
468 |                 x += width + 20
469 |                 row = 0  # Reset row counter for new column
470 |                 
471 |         return y
472 | 
473 |     # Fourth page - Feature Importance
474 |     chm_outputs_dir = os.path.dirname(os.path.dirname(output_dir))  # Get chm_outputs directory
475 |     model_eval_path = get_latest_file(chm_outputs_dir, 'model_evaluation', required=False)
476 |     if model_eval_path:
477 |         try:
478 |             with open(model_eval_path) as f:
479 |                 model_data = json.load(f)
480 |             
481 |             if 'feature_importance' in model_data:
482 |                 # Add title
483 |                 c.setFont("Helvetica-Bold", 14)
484 |                 # c.drawString(50, height-40, "Feature Importance Analysis")
485 |                 
486 |                 # # Add chart
487 |                 # chart = create_feature_importance_chart(model_data['feature_importance'], width-100, height/2)
488 |                 # chart.drawOn(c, 50, height/2)
489 |                 
490 |                 # Add table below chart
491 |                 table_y = height/2 - 20
492 |                 c.setFont("Helvetica-Bold", 12)
493 |                 c.drawString(50, table_y, "Feature Importance Values")
494 |                 table_y -= 20
495 |                 draw_feature_importance_table(c, model_data['feature_importance'], 50, table_y, width/2-70)
496 |                 
497 |                 c.showPage()
498 |         except Exception as e:
499 |             print(f"Warning: Could not load feature importance data: {e}")
500 |     # if model_eval_path and os.path.exists(model_eval_path):
501 |     #     try:
502 |     #         with open(model_eval_path) as f:
503 |     #             model_data = json.load(f)
504 |             
505 |     #         if 'feature_importance' in model_data:
506 |     #             c.setFont("Helvetica-Bold", 14)
507 |     #             c.drawString(50, height-40, "Feature Importance Analysis")
508 |                 
509 |     #             # Create and add feature importance chart
510 |     #             chart = create_feature_importance_chart(model_data['feature_importance'], width-100, height-100)
511 |     #             chart.drawOn(c, 50, 50)
512 |                 
513 |     #             c.showPage()
514 |     #     except Exception as e:
515 |     #         print(f"Warning: Could not load feature importance data: {e}")
516 |     
517 |     c.save()
518 |     return pdf_path


--------------------------------------------------------------------------------
/sentinel1_source.py:
--------------------------------------------------------------------------------
  1 | import ee
  2 | from typing import Union, List
  3 | import math
  4 | 
  5 | # def mask_border_noise(img):
  6 | #     """Border Noise Correction"""
  7 | #     # Check if mask band exists
  8 | #     bands = img.bandNames().getInfo()
  9 | #     if 'mask' in bands:
 10 | #         mask = img.select(['mask']).eq(0)  # Mask border noise pixels
 11 | #         return img.updateMask(mask).copyProperties(img, ['system:time_start'])
 12 | #     return img.copyProperties(img, ['system:time_start'])
 13 | 
 14 | def gamma_map_filter(image, kernel_size, enl):
 15 |     """Speckle Filter - Gamma Map Multi-temporal"""
 16 |     bands = ['VV', 'VH']
 17 |     
 18 |     def filter_band(band):
 19 |         mean = image.select([band]).reduceNeighborhood(
 20 |             reducer=ee.Reducer.mean(),
 21 |             kernel=ee.Kernel.square(kernel_size/2, 'pixels')
 22 |         )
 23 |         variance = image.select([band]).reduceNeighborhood(
 24 |             reducer=ee.Reducer.variance(),
 25 |             kernel=ee.Kernel.square(kernel_size/2, 'pixels')
 26 |         )
 27 |         # Convert to numbers for mathematical operations
 28 |         mean_num = mean.toFloat()
 29 |         variance_num = variance.toFloat()
 30 |         enl_num = ee.Number(enl)
 31 |         
 32 |         # Calculate coefficient of variation
 33 |         cv = variance_num.divide(mean_num.pow(2)).sqrt()
 34 |         
 35 |         # Calculate weight
 36 |         weight = cv.pow(-2).divide(cv.pow(-2).add(enl_num))
 37 |         
 38 |         # Apply filter
 39 |         return mean_num.multiply(weight).add(
 40 |             image.select([band]).toFloat().multiply(ee.Number(1).subtract(weight))
 41 |         )
 42 |     
 43 |     filtered = ee.Image(ee.List(bands).map(filter_band))
 44 |     return image.addBands(filtered.rename(bands), None, True)
 45 | 
 46 | def terrain_flattening(img, dem):
 47 |     """Terrain Flattening"""
 48 |     dem = ee.Image(dem)
 49 |     theta_i = img.select(['angle']).toFloat().multiply(ee.Number(math.pi/180))
 50 |     slope = ee.Terrain.slope(dem).toFloat().multiply(ee.Number(math.pi/180))
 51 |     aspect = ee.Terrain.aspect(dem).toFloat().multiply(ee.Number(math.pi/180))
 52 |     
 53 |     # Calculate projection angle
 54 |     phi_i = ee.Algorithms.If(
 55 |         ee.String(img.get('orbitProperties_pass')).equals('ASCENDING'),
 56 |         ee.Number(0),
 57 |         ee.Number(math.pi)
 58 |     )
 59 |     
 60 |     # Calculate d_phi using image operations
 61 |     d_phi = aspect.subtract(ee.Image.constant(phi_i))
 62 |     
 63 |     # Calculate cos_theta_s
 64 |     cos_theta_s = theta_i.cos().multiply(slope.cos()) \
 65 |         .add(theta_i.sin().multiply(slope.sin()).multiply(d_phi.cos()))
 66 |     theta_s = cos_theta_s.acos()
 67 |     
 68 |     # Apply volume model correction
 69 |     correction = theta_s.sin().divide(theta_i.sin())
 70 |     
 71 |     # Select and apply correction
 72 |     return img.select(['VV', 'VH']).toFloat() \
 73 |         .multiply(correction) \
 74 |         .set('system:time_start', img.get('system:time_start'))
 75 | 
 76 | def get_sentinel1_data(
 77 |     aoi: ee.Geometry,
 78 |     year: int,
 79 |     start_date: str,
 80 |     end_date: str
 81 | ) -> ee.Image:
 82 |     """
 83 |     Get Sentinel-1 data for the specified area and time period.
 84 |     
 85 |     Args:
 86 |         aoi: Area of interest as Earth Engine Geometry
 87 |         year: Year for analysis
 88 |         start_date: Start date for Sentinel-1 data
 89 |         end_date: End date for Sentinel-1 data
 90 |     
 91 |     Returns:
 92 |         ee.Image: Processed Sentinel-1 data
 93 |     """
 94 |     # Import Sentinel-1 dataset
 95 |     s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
 96 |     
 97 |     # Filter by date and region
 98 |     s1_filtered = s1.filterDate(f"{year}-{start_date}", f"{year}-{end_date}") \
 99 |                     .filterBounds(aoi)
100 |     
101 |     # Filter by instrument mode and polarization
102 |     s1_filtered = s1_filtered.filter(ee.Filter.eq('instrumentMode', 'IW')) \
103 |                             .filter(ee.Filter.eq('resolution_meters', 10)) \
104 |                             .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
105 |                             .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
106 |     
107 |     # Select VV and VH bands
108 |     s1_filtered = s1_filtered.select(['VV', 'VH', 'angle'])
109 |     
110 |     # Processing chain
111 |     s1_processed = s1_filtered \
112 |         # .map(lambda img: gamma_map_filter(img, 15, 10)) \
113 |         # .map(lambda img: terrain_flattening(img, 'USGS/SRTMGL1_003'))
114 |         # .map(mask_border_noise) \
115 |     
116 |     # Calculate temporal statistics
117 |     s1_median = s1_processed.select(['VV', 'VH'],['S1_VV','S1_VH']).median()
118 |     s1_median = s1_median.clip(aoi)
119 |     
120 |     return s1_median 


--------------------------------------------------------------------------------
/sentinel2_source.py:
--------------------------------------------------------------------------------
  1 | import ee
  2 | from typing import Union, List
  3 | 
  4 | def get_sentinel2_data(
  5 |     aoi: ee.Geometry,
  6 |     year: int,
  7 |     start_date: str,
  8 |     end_date: str,
  9 |     clouds_th: int,
 10 | ) -> ee.Image:
 11 |     """
 12 |     Get Sentinel-2 data for the specified area and time period.
 13 |     
 14 |     Args:
 15 |         aoi: Area of interest as Earth Engine Geometry
 16 |         year: Year for analysis
 17 |         start_date: Start date for Sentinel-2 data
 18 |         end_date: End date for Sentinel-2 data
 19 |         clouds_th: Cloud threshold (0-100)
 20 |     
 21 |     Returns:
 22 |         ee.Image: Processed Sentinel-2 data
 23 |     """
 24 |     # Format dates properly for Earth Engine
 25 |     start_date_ee = ee.Date(f'{year}-{start_date}')
 26 |     end_date_ee = ee.Date(f'{year}-{end_date}')
 27 |     
 28 |     # Import Sentinel-2 dataset
 29 |     s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') #original code used 'COPERNICUS/S2_SR_HARMONIZED')
 30 |     
 31 |     # Filter by date and region
 32 |     s2_filtered = s2.filterDate(start_date_ee, end_date_ee) \
 33 |                     .filterBounds(aoi) #\
 34 |                     # .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', clouds_th))
 35 |     
 36 |     # Get cloud probability data
 37 |     S2_CLOUD_PROBABILITY = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
 38 |         .filterDate(start_date_ee, end_date_ee) \
 39 |         .filterBounds(aoi)
 40 |     
 41 |     s2_filtered = ee.ImageCollection(s2_filtered) \
 42 |                     .map(lambda img: img.addBands(S2_CLOUD_PROBABILITY.filter(ee.Filter.equals('system:index', img.get('system:index'))).first()))
 43 |     # For S2, get cloud probability called 'cloud_mask' from the S2_CLOUD_PROBABILITY collection without joining
 44 |     # s2_filtered = ee.ImageCollection(s2_filtered).merge(S2_CLOUD_PROBABILITY.select('cloud_mask'))
 45 |         
 46 |     # # Join with cloud probability data
 47 |     # join_filter = ee.Filter.And(
 48 |     #     ee.Filter.equals('system:index', 'system:index'),
 49 |     #     ee.Filter.equals('system:time_start', 'system:time_start')
 50 |     # )
 51 |     
 52 |     # s2_filtered = ee.Join.saveFirst('cloud_mask').apply(
 53 |     #     primary=s2_filtered,
 54 |     #     secondary=S2_CLOUD_PROBABILITY,
 55 |     #     condition=join_filter
 56 |     # )
 57 |     
 58 |     # s2_filtered = ee.ImageCollection(s2_filtered)
 59 |     
 60 |     def maskClouds(img):
 61 |         clouds = ee.Image(img).select('probability')
 62 |         # ee.Image(img.get('cloud_mask')).select('probability')
 63 |         isNotCloud = clouds.lt(clouds_th)
 64 |         return img.mask(isNotCloud)
 65 |     
 66 |     def maskEdges(s2_img):
 67 |         return s2_img.updateMask(
 68 |             s2_img.select('B8A').mask().updateMask(s2_img.select('B9').mask())
 69 |         )#.updateMask(mask_raster.eq(1))
 70 | 
 71 |     s2_filtered = s2_filtered.map(maskEdges) 
 72 |     s2_filtered = s2_filtered.map(maskClouds) 
 73 |     
 74 |     # Select relevant bands
 75 |     s2_filtered = s2_filtered.select([
 76 |         'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'
 77 |     ])
 78 |     
 79 |     # Calculate vegetation indices
 80 |     def add_indices(img):
 81 |         # Normalized Difference Vegetation Index (NDVI)
 82 |         ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
 83 |         
 84 |         # # Enhanced Vegetation Index (EVI)
 85 |         # evi = img.expression(
 86 |         #     '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
 87 |         #     {
 88 |         #         'NIR': img.select('B8'),
 89 |         #         'RED': img.select('B4'),
 90 |         #         'BLUE': img.select('B2')
 91 |         #     }
 92 |         # ).rename('EVI')
 93 |         
 94 |         # # Normalized Difference Water Index (NDWI)
 95 |         # ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI')
 96 |         
 97 |         # # Normalized Difference Moisture Index (NDMI)
 98 |         # ndmi = img.normalizedDifference(['B8', 'B11']).rename('NDMI')
 99 |         
100 |         # return img.addBands([ndvi, evi, ndwi, ndmi])
101 |         return img.addBands([ndvi])
102 | 
103 |     
104 |     s2_filtered = s2_filtered.map(add_indices)
105 |     
106 |     # Calculate temporal statistics
107 |     s2_processed = s2_filtered.median()
108 |     s2_processed = s2_processed.clip(aoi)
109 |     # s2_processed = s2_processed.clip(geometry)
110 |     
111 |     return s2_processed 


--------------------------------------------------------------------------------
/train_predict_map.py:
--------------------------------------------------------------------------------
  1 | import numpy as np
  2 | import pandas as pd
  3 | from sklearn.ensemble import RandomForestRegressor
  4 | from sklearn.model_selection import train_test_split
  5 | import torch
  6 | import torch.optim as optim
  7 | import torch.nn as nn
  8 | from dl_models import MLPRegressionModel, create_normalized_dataloader
  9 | import rasterio
 10 | from rasterio.mask import geometry_mask
 11 | from shapely.geometry import Point, box
 12 | from shapely.ops import transform
 13 | import geopandas as gpd
 14 | import os
 15 | from pathlib import Path
 16 | from typing import Tuple, Optional
 17 | import warnings
 18 | import argparse
 19 | from tqdm import tqdm
 20 | warnings.filterwarnings('ignore')
 21 | 
 22 | from evaluate_predictions import calculate_metrics
 23 | 
 24 | def load_training_data(csv_path: str, mask_path: Optional[str] = None,
 25 |                       feature_names: Optional[list] = None, ch_col: str = 'rh') -> Tuple[np.ndarray, np.ndarray]:
 26 |     """
 27 |     Load training data from CSV file and optionally mask with forest mask.
 28 |     
 29 |     Args:
 30 |         csv_path: Path to training data CSV
 31 |         mask_path: Optional path to forest mask TIF
 32 |         
 33 |     Returns:
 34 |         X: Feature matrix
 35 |         y: Target variable (rh)
 36 |     """
 37 |     # Read training data
 38 |     df = pd.read_csv(csv_path)
 39 |     
 40 |     # Create GeoDataFrame from points
 41 |     gdf = gpd.GeoDataFrame(
 42 |         df,
 43 |         geometry=[Point(xy) for xy in zip(df['longitude'], df['latitude'])],
 44 |         crs="EPSG:4326"
 45 |     )
 46 |     
 47 |     if mask_path:
 48 |         with rasterio.open(mask_path) as mask_src:
 49 |             # Check CRS
 50 |             mask_crs = mask_src.crs
 51 |             if mask_crs != gdf.crs:
 52 |                 gdf = gdf.to_crs(mask_crs)
 53 |             
 54 |             # Get bounds of mask
 55 |             mask_bounds = box(*mask_src.bounds)
 56 |             
 57 |             # First filter points by mask bounds
 58 |             gdf_masked = gdf[gdf.geometry.within(mask_bounds)]
 59 |             
 60 |             if len(gdf_masked) == 0:
 61 |                 raise ValueError("No training points fall within the mask bounds")
 62 |             else:
 63 |                 gdf = gdf_masked
 64 |             
 65 |             # Convert points to pixel coordinates
 66 |             pts_pixels = []
 67 |             valid_indices = []
 68 |             for idx, point in enumerate(gdf.geometry):
 69 |                 row, col = rasterio.transform.rowcol(mask_src.transform, 
 70 |                                                    point.x, 
 71 |                                                    point.y)
 72 |                 if (0 <= row < mask_src.height and 
 73 |                     0 <= col < mask_src.width):
 74 |                     pts_pixels.append((row, col))
 75 |                     valid_indices.append(idx)
 76 |             
 77 |             if not pts_pixels:
 78 |                 raise ValueError("No training points could be mapped to valid pixels")
 79 |             
 80 |             # Read forest mask values at pixel locations
 81 |             mask_values = [mask_src.read(1)[r, c] for r, c in pts_pixels]
 82 |             
 83 |             # Filter points by mask values
 84 |             mask_indices = [i for i, v in enumerate(mask_values) if v == 1]
 85 |             if not mask_indices:
 86 |                 raise ValueError("No training points fall within the forest mask")
 87 |             
 88 |             final_indices = [valid_indices[i] for i in mask_indices]
 89 |             gdf = gdf.iloc[final_indices]
 90 |     
 91 |     # Convert back to original CRS if needed
 92 |     if mask_path and mask_crs != "EPSG:4326":
 93 |         gdf = gdf.to_crs("EPSG:4326")
 94 |     
 95 |     # Separate features and target
 96 |     df = pd.DataFrame(gdf.drop(columns='geometry'))
 97 |     y = df[ch_col].values
 98 |     
 99 |     # Get feature columns in same order as feature_names
100 |     if feature_names is not None:
101 |         missing_features = set(feature_names) - set(df.columns)
102 |         if missing_features:
103 |             raise ValueError(f"Missing features in training data: {missing_features}")
104 |         X = df[feature_names].values
105 |     else:
106 |         X = df.drop([ch_col, 'longitude', 'latitude'], axis=1, errors='ignore').values
107 |     
108 |     return X, y
109 | 
110 | def load_prediction_data(stack_path: str, mask_path: Optional[str] = None, feature_names: Optional[list] = None) -> Tuple[np.ndarray, rasterio.DatasetReader]:
111 |     """
112 |     Load prediction data from stack TIF and optionally apply forest mask.
113 |     
114 |     Args:
115 |         stack_path: Path to stack TIF file
116 |         mask_path: Optional path to forest mask TIF
117 |         feature_names: Optional list of feature names for filtering bands
118 |         
119 |     Returns:
120 |         X: Feature matrix for prediction
121 |         src: Rasterio dataset for writing results
122 |     """
123 |     if feature_names is None:
124 |         raise ValueError("feature_names must be provided to ensure consistent features between training and prediction")
125 |     # Read stack file
126 |     with rasterio.open(stack_path) as src:
127 |         stack = src.read()
128 |         stack_crs = src.crs
129 |         
130 |         # Get band descriptions if available
131 |         band_descriptions = src.descriptions
132 |         
133 |         # Filter bands based on feature names if provided
134 |         # Create a mapping of band descriptions to indices
135 |         band_indices = []
136 |         for i, desc in enumerate(band_descriptions):
137 |             if desc in feature_names:
138 |                 band_indices.append(i)
139 |         
140 |         if len(band_indices) != len(feature_names):
141 |             missing_features = set(feature_names) - set(band_descriptions)
142 |             raise ValueError(f"Could not find all feature names in stack bands. Missing features: {missing_features}")
143 |         
144 |         # Select only the bands that match feature names
145 |         stack = stack[band_indices]
146 |         
147 |         # Reshape stack to 2D array (bands x pixels)
148 |         n_bands, height, width = stack.shape
149 |         X = stack.reshape(n_bands, -1).T
150 |         
151 |         # Apply mask if provided
152 |         if mask_path:
153 |             with rasterio.open(mask_path) as mask_src:
154 |                 # Check CRS
155 |                 if mask_src.crs != stack_crs:
156 |                     raise ValueError(f"CRS mismatch: stack {stack_crs} != mask {mask_src.crs}")
157 |                 
158 |                 # Check dimensions
159 |                 if mask_src.shape != (height, width):
160 |                     raise ValueError(f"Shape mismatch: stack {(height, width)} != mask {mask_src.shape}")
161 |                 
162 |                 mask = mask_src.read(1)
163 |                 mask = mask.reshape(-1)
164 |                 X = X[mask == 1]
165 |         
166 |         src_copy = rasterio.open(stack_path)
167 |         return X, src_copy
168 | 
169 | def save_metrics_and_importance(metrics: dict, importance_data: dict, output_dir: str) -> None:
170 |     """
171 |     Save training metrics and feature importance to JSON file, ensuring all values are JSON serializable.
172 |     """
173 |     # Convert any non-serializable values to Python native types
174 |     serializable_metrics = {}
175 |     for key, value in metrics.items():
176 |         if hasattr(value, 'item'):  # Handle numpy/torch numbers
177 |             serializable_metrics[key] = value.item()
178 |         else:
179 |             serializable_metrics[key] = float(value)
180 |     
181 |     serializable_importance = {}
182 |     for key, value in importance_data.items():
183 |         if hasattr(value, 'item'):  # Handle numpy/torch numbers
184 |             serializable_importance[key] = value.item()
185 |         else:
186 |             serializable_importance[key] = float(value)
187 |     """
188 |     Save training metrics and feature importance to JSON file.
189 |     
190 |     Args:
191 |         metrics: Dictionary of training metrics
192 |         importance_data: Dictionary of feature importance data
193 |         output_dir: Directory to save JSON file
194 |     """
195 |     import json
196 |     from pathlib import Path
197 |     
198 |     # Combine metrics and importance data
199 |     output_data = {
200 |         "train_metrics": serializable_metrics,
201 |         "feature_importance": serializable_importance
202 |     }
203 |     
204 |     # Create output path
205 |     output_path = Path(output_dir) / "model_evaluation.json"
206 |     
207 |     # Save to JSON
208 |     with open(output_path, "w") as f:
209 |         json.dump(output_data, f, indent=4)
210 |     print(f"Saved model evaluation data to: {output_path}")
211 | 
212 | def train_model(X: np.ndarray, y: np.ndarray, model_type: str = 'rf', batch_size: int = 64,
213 |                 test_size: float = 0.2, feature_names: Optional[list] = None,
214 |                 n_bands: Optional[int] = None) -> Tuple[object, dict, dict]:
215 |     """
216 |     Train model with optional validation split.
217 |     
218 |     Args:
219 |         X: Feature matrix
220 |         y: Target variable
221 |         model_type: Type of model ('rf' or 'mlp')
222 |         batch_size: Batch size for MLP training
223 |         test_size: Proportion of data to use for validation
224 |         feature_names: Optional list of feature names
225 |         
226 |     Returns:
227 |         Trained model, training metrics, and feature importance/weights
228 |     """
229 |     # Split data
230 |     X_train, X_val, y_train, y_val = train_test_split(
231 |         X, y, test_size=test_size, random_state=42
232 |     )
233 |     
234 |     if model_type == 'rf':
235 |         # Train Random Forest model
236 |         model = RandomForestRegressor(
237 |             n_estimators=500,
238 |             min_samples_leaf=5,
239 |             max_features='sqrt',
240 |             n_jobs=-1,
241 |             random_state=42
242 |         )
243 |         model.fit(X_train, y_train)
244 |         
245 |         # Get predictions
246 |         y_pred = model.predict(X_val)
247 |         train_metrics = calculate_metrics(y_pred, y_val)
248 |         
249 |         # Get feature importance
250 |         importance = model.feature_importances_
251 |         if feature_names is None:
252 |             feature_names = [f"feature_{i}" for i in range(len(importance))]
253 |         
254 |         importance_data = {
255 |             name: float(imp) for name, imp in zip(feature_names, importance)
256 |         }
257 |         
258 |     else:  # MLP model
259 |         # Create normalized dataloaders
260 |         train_loader, val_loader, scaler_mean, scaler_std = create_normalized_dataloader(
261 |             X_train, X_val, y_train, y_val, batch_size=batch_size, n_bands=n_bands
262 |         )
263 |         
264 |         # Initialize model
265 |         model = MLPRegressionModel(input_size=X.shape[1])
266 |         if torch.cuda.is_available():
267 |             model = model.cuda()
268 |             
269 |         # Training setup
270 |         criterion = nn.MSELoss()
271 |         optimizer = optim.Adam(model.parameters())
272 |         num_epochs = 100
273 |         best_val_loss = float('inf')
274 |         
275 |         # Training loop with tqdm progress bar
276 |         for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
277 |             model.train()
278 |             for batch_X, batch_y in train_loader:
279 |                 if torch.cuda.is_available():
280 |                     batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
281 |                 
282 |                 optimizer.zero_grad()
283 |                 outputs = model(batch_X)
284 |                 loss = criterion(outputs, batch_y)
285 |                 loss.backward()
286 |                 optimizer.step()
287 |             
288 |             # Validation
289 |             model.eval()
290 |             val_predictions = []
291 |             val_targets = []
292 |             with torch.no_grad():
293 |                 for batch_X, batch_y in val_loader:
294 |                     if torch.cuda.is_available():
295 |                         batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
296 |                     outputs = model(batch_X)
297 |                     val_predictions.extend(outputs.cpu().numpy())
298 |                     val_targets.extend(batch_y.cpu().numpy())
299 |             
300 |             val_predictions = np.array(val_predictions)
301 |             val_targets = np.array(val_targets)
302 |             val_metrics = calculate_metrics(val_predictions, val_targets)
303 |             val_loss = val_metrics['RMSE']
304 |             
305 |             if val_loss < best_val_loss:
306 |                 best_val_loss = val_loss
307 |                 train_metrics = val_metrics
308 |         
309 |         # Get feature importance (using weights of first layer as proxy)
310 |         with torch.no_grad():
311 |             weights = model.layers[0].weight.abs().mean(dim=0).cpu().numpy()
312 |             if feature_names is None:
313 |                 feature_names = [f"feature_{i}" for i in range(len(weights))]
314 |             importance_data = {}
315 |             for name, weight in zip(feature_names, weights):
316 |                 # Convert numpy.float32 to Python float
317 |                 weight_value = weight.item() if hasattr(weight, 'item') else float(weight)
318 |                 importance_data[name] = weight_value
319 |         
320 |         # Store normalization parameters with model
321 |         model.scaler_mean = scaler_mean
322 |         model.scaler_std = scaler_std
323 |     
324 |     # Sort importance by value
325 |     importance_data = dict(sorted(importance_data.items(), key=lambda x: x[1], reverse=True))
326 |     
327 |     # Print metrics and top features
328 |     for metric, value in train_metrics.items():
329 |         print(f"{metric}: {value:.3f}")
330 |     
331 |     print("\nTop 5 Important Features:")
332 |     for name, imp in list(importance_data.items())[:5]:
333 |         print(f"{name}: {imp:.3f}")
334 |     
335 |     return model, train_metrics, importance_data
336 | 
337 | def save_predictions(predictions: np.ndarray, src: rasterio.DatasetReader, output_path: str,
338 |                     mask_path: Optional[str] = None) -> None:
339 |     """
340 |     Save predictions to a GeoTIFF file.
341 |     
342 |     Args:
343 |         predictions: Model predictions
344 |         src: Source rasterio dataset for metadata
345 |         output_path: Path to save predictions
346 |         mask_path: Optional path to forest mask TIF
347 |     """
348 |     # Create output profile
349 |     profile = src.profile.copy()
350 |     profile.update(count=1, dtype='float32')
351 |     
352 |     # Initialize prediction array
353 |     height, width = src.height, src.width
354 |     pred_array = np.zeros((height, width), dtype='float32')
355 |     
356 |     if mask_path:
357 |         # Apply predictions only to masked areas
358 |         with rasterio.open(mask_path) as mask_src:
359 |             # Check CRS
360 |             if mask_src.crs != src.crs:
361 |                 raise ValueError(f"CRS mismatch: source {src.crs} != mask {mask_src.crs}")
362 |             
363 |             mask = mask_src.read(1)
364 |             mask_idx = np.where(mask.reshape(-1) == 1)[0]
365 |             pred_array.reshape(-1)[mask_idx] = predictions
366 |     else:
367 |         # Apply predictions to all pixels
368 |         pred_array = predictions.reshape(height, width)
369 |     
370 |     try:
371 |         # Save predictions
372 |         with rasterio.open(output_path, 'w', **profile) as dst:
373 |             dst.write(pred_array, 1)
374 |     finally:
375 |         src.close()
376 | 
377 | def parse_args():
378 |     parser = argparse.ArgumentParser(description='Train model and generate canopy height predictions')
379 |     
380 |     # Input paths
381 |     parser.add_argument('--training-data', type=str, required=True,
382 |                        help='Path to training data CSV')
383 |     parser.add_argument('--stack', type=str, required=True,
384 |                        help='Path to stack TIF file')
385 |     parser.add_argument('--mask', type=str, required=True,
386 |                        help='Path to forest mask TIF')
387 |     parser.add_argument('--buffered-mask', type=str, required=True,
388 |                        help='Path to buffered forest mask TIF')
389 |     # Output settings
390 |     parser.add_argument('--output-dir', type=str, default='chm_outputs',
391 |                        help='Output directory for predictions')
392 |     # parser.add_argument('--output-filename', type=str, default='canopy_height_predictions.tif',
393 |     #                    help='Output filename for predictions')
394 |     
395 |     # Model parameters
396 |     parser.add_argument('--model', type=str, default='rf', choices=['rf', 'mlp'],
397 |                        help='Model type: random forest (rf) or MLP neural network (mlp)')
398 |     parser.add_argument('--batch-size', type=int, default=64,
399 |                        help='Batch size for MLP training')
400 |     parser.add_argument('--n-bands', type=int, default=None,
401 |                        help='Number of spectral bands for band-wise normalization')
402 |     parser.add_argument('--test-size', type=float, default=0.2,
403 |                        help='Proportion of data to use for validation')
404 |     parser.add_argument('--apply-forest-mask', action='store_true',
405 |                        help='Apply forest mask to predictions')
406 |     parser.add_argument('--ch_col', type=str, default='rh',
407 |                        help='Column name for canopy height')
408 |     return parser.parse_args()
409 | 
410 | def main():
411 |     # Parse arguments
412 |     args = parse_args()
413 |     
414 |     ch_col = args.ch_col
415 |     # Create output directory
416 |     os.makedirs(args.output_dir, exist_ok=True)
417 |     
418 |     # Load training data
419 |     print("Loading training data...")
420 |     df = pd.read_csv(args.training_data)
421 |     
422 |     # Filter out samples with slope > 20
423 |     slope_cols = [col for col in df.columns if col.endswith('_slope')]
424 |     for col in slope_cols:
425 |         df = df[df[col] <= 20]
426 |     
427 |     # Define columns to remove
428 |     remove_cols = [ch_col, 'longitude', 'latitude', 'rh',
429 |                   'digital_elevation_model', 'digital_elevation_model_srtm', 'elev_lowestmode']
430 |     feature_names = [col for col in df.columns if col not in remove_cols]
431 |     
432 |     # Write filtered DataFrame back to file for consistency
433 |     filtered_training_path = os.path.join(args.output_dir, 'filtered_training.csv')
434 |     df.to_csv(filtered_training_path, index=False)
435 |     
436 |     # Load filtered training data
437 |     X, y = load_training_data(filtered_training_path, args.buffered_mask,
438 |                              feature_names=feature_names, ch_col=ch_col)
439 |     print(f"Loaded training data with {X.shape[1]} features and {len(y)} samples")
440 |     
441 |     # Train model
442 |     print("Training model...")
443 |     model, train_metrics, importance_data = train_model(
444 |         X, y,
445 |         model_type=args.model,
446 |         batch_size=args.batch_size,
447 |         test_size=args.test_size,
448 |         feature_names=feature_names,
449 |         n_bands=args.n_bands
450 |     )
451 |     
452 |     # Save metrics and importance
453 |     save_metrics_and_importance(train_metrics, importance_data, args.output_dir)
454 |     
455 |     # Load prediction data
456 |     print("Loading prediction data...")
457 |     print(f"Using {len(feature_names)} features for prediction: {', '.join(feature_names)}")
458 |     X_pred, src = load_prediction_data(args.stack, args.mask, feature_names=feature_names)
459 |     print(f"Loaded prediction data with shape: {X_pred.shape}")
460 |     
461 |     if X_pred.shape[1] != X.shape[1]:
462 |         raise ValueError(f"Feature count mismatch: Training has {X.shape[1]} features, but prediction data has {X_pred.shape[1]} features")
463 |     
464 |     # Make predictions
465 |     print("Generating predictions...")
466 |     if args.model == 'rf':
467 |         predictions = model.predict(X_pred)
468 |     else:  # MLP model
469 |         model.eval()
470 |         with torch.no_grad():
471 |             # Normalize prediction data
472 |             X_pred_tensor = torch.FloatTensor(X_pred)
473 |             X_pred_normalized = (X_pred_tensor - model.scaler_mean) / model.scaler_std
474 |             
475 |             # Make predictions in batches
476 |             predictions = []
477 |             for i in range(0, len(X_pred), args.batch_size):
478 |                 batch = X_pred_normalized[i:i + args.batch_size]
479 |                 if torch.cuda.is_available():
480 |                     batch = batch.cuda()
481 |                 pred = model(batch)
482 |                 predictions.extend(pred.cpu().numpy())
483 |             predictions = np.array(predictions)
484 |     print(f"Generated {len(predictions)} predictions")
485 |     output_path = Path(args.output_dir) / f"{Path(args.stack).stem.replace('stack_', 'predictCH')}_{ch_col}.tif"
486 |     
487 |     # Save predictions
488 |     # output_path = os.path.join(args.output_dir, output_filename)
489 |     print(f"Saving predictions to: {output_path}")
490 |     save_predictions(predictions, src, output_path, args.mask)
491 |     print("Done!")
492 | 
493 | if __name__ == "__main__":
494 |     main()


--------------------------------------------------------------------------------
/utils.py:
--------------------------------------------------------------------------------
 1 | import os
 2 | 
 3 | def get_latest_file(dir_path: str, pattern: str, required: bool = True) -> str:
 4 |     files = [f for f in os.listdir(dir_path) if f.startswith(pattern)]
 5 |     if not files:
 6 |         if required:
 7 |             raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {dir_path}")
 8 |         return None
 9 |     return os.path.join(dir_path, max(files, key=lambda x: os.path.getmtime(os.path.join(dir_path, x))))
10 | 


--------------------------------------------------------------------------------