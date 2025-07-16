# Height Correlation Analysis Plan

## Objective
Analyze correlations between reference heights (from downloads folder) and auxiliary height data (GEDI rh and canopy height data) from the embedding patch dataset to understand data quality and relationships across the three study regions.

## Data Sources

### 1. Reference Heights
- **Location**: `downloads/dchm_*.tif` files
- **Regions**: 
  - Kochi: `dchm_04hf3.tif`
  - Hyogo: `dchm_05LE4.tif` 
  - Tochigi: `dchm_09gd4.tif`
- **Description**: Ground truth canopy height measurements

### 2. Auxiliary Height Data (from embedding patches)
Based on `chm_main.py`, the following auxiliary height data sources are available:

#### a) GEDI Data
- **Function**: `get_gedi_vector_data()` from `l2a_gedi_source.py`
- **Band Name**: `rh` (from `args.quantile`, default='rh98')
- **Characteristics**: Sparse point data, converted to raster via `reduceToImage()`
- **Data Type**: LIDAR-based height measurements
- **Note**: Some patches may not have GEDI data

#### b) Canopy Height Products
- **Function**: `get_canopyht_data()` from `canopyht_source.py`  
- **Band Names**: 
  - `ch_pauls2024`: Paul's 2024 canopy height product
  - `ch_tolan2024`: Tolan 2024 canopy height product  
  - `ch_lang2022`: Lang 2022 canopy height product
  - `ch_potapov2021`: Potapov 2021 canopy height product
- **Characteristics**: Different satellite-derived height products
- **Data Type**: Global canopy height estimates from various methodologies

## Analysis Approach

### 1. Data Extraction
- Extract reference height data from downloads TIF files
- Extract auxiliary height data from embedding patches (bands 65-70)
- Spatially align data using common coordinate system
- Handle sparse GEDI data by analyzing only valid pixels
- **Filter zero reference heights**: Exclude pixels where reference height = 0 to avoid bias from non-forest areas
- **Apply forest mask**: Use forest_mask band to exclude non-forest pixels (forest_mask = 1 for forest, 0 for non-forest)

### 2. Regional Analysis
For each region (Kochi, Hyogo, Tochigi):
- Compare reference heights vs GEDI heights (where available)
- Compare reference heights vs each canopy height product:
  - Paul's 2024 (`ch_pauls2024`)
  - Tolan 2024 (`ch_tolan2024`) 
  - Lang 2022 (`ch_lang2022`)
  - Potapov 2021 (`ch_potapov2021`)
- Calculate correlation metrics: R², RMSE, bias
- Generate scatter plots with regression lines

### 3. Metrics Calculation
- **R² (Coefficient of Determination)**: Measure of linear relationship strength
- **RMSE (Root Mean Square Error)**: Measure of prediction accuracy
- **Bias**: Mean difference between auxiliary and reference heights
- **Sample Size**: Number of valid comparison pixels

### 4. Visualization
- Scatter plots with 1:1 reference line
- Linear regression fitting
- Color-coded by density if needed
- Statistical metrics displayed on plots
- Separate plots for each region and height source

## Expected Outputs

### 1. Correlation Plots
- **Individual plots**: Up to 15 plots total (3 regions × 5 auxiliary height sources)
- **Combined plot**: Single PNG with 3×5 grid (3 regions rows × 5 height sources columns)
- **Format**: PNG files with high resolution and hexbin density visualization
- **Location**: `chm_outputs/plot_analysis/`

### 2. Statistical Summary
- CSV file with correlation metrics per region
- Location: `chm_outputs/plot_analysis/height_correlation_summary.csv`

### 3. Analysis Report
- Markdown document with findings
- Location: `docs/height_correlation_results.md`

## Implementation Steps

1. **Data Loading**: Load reference TIFs and embedding patches
2. **Band Identification**: Identify auxiliary height bands in patches
3. **Spatial Registration**: Align data to common grid
4. **Correlation Analysis**: Calculate metrics for valid pixel pairs
5. **Visualization**: Generate correlation plots
6. **Documentation**: Update CLAUDE.md with results

## Technical Considerations

### Data Handling
- Use rasterio for TIF file operations
- Handle different spatial resolutions (10m patches vs reference resolution)
- Account for NoData values and missing pixels
- **Zero height filtering**: Remove pixels where reference height = 0 to focus on forested areas
- **Forest mask filtering**: Use forest_mask band to exclude non-forest pixels (only analyze where forest_mask = 1)
- Use spatial resampling if needed

### GEDI Data Specifics
- GEDI data is sparse - only analyze pixels with valid GEDI values
- **Patch selection**: Use patches with highest GEDI point density per region
  - Kochi (`dchm_04hf3`): Patches 0 (130 pixels)
  - Hyogo (`dchm_05LE4`): Patches 33, 27, 13 (1118, 977, 973 pixels)
  - Tochigi (`dchm_09gd4`): Patches 42, 49, 12 (128, 102, 89 pixels)
- **Multiple patches**: Analyze up to 3 patches per region for comprehensive coverage
- **Outlier filtering**: Apply same outlier mask to GEDI data as other height sources
- May need to buffer GEDI points for better spatial matching
- Consider uncertainty in GEDI geolocation

### Performance
- Process patches in batches to manage memory
- Use efficient spatial operations
- Cache intermediate results if needed

## Success Criteria
- Correlation analysis completed for all three regions
- Statistical significance of relationships determined
- Clear visualization of height data relationships
- Documentation updated with key findings
- Identification of best auxiliary height source per region

### Initial Correlation Results

The initial correlation analysis, using a few sample patches per region, revealed poor performance across all auxiliary height sources. The following table summarizes the key metrics:

| Region  | Height Source | R²     | RMSE (m) | Bias (m) | Sample Size |
|---------|---------------|--------|----------|----------|-------------|
| Kochi   | GEDI          | -1.52  | 8.50     | 3.20     | 32          |
| Kochi   | Pauls2024     | -2.55  | 10.67    | 8.92     | 21,079      |
| Kochi   | Tolan2024     | -1.19  | 8.18     | -5.48    | 20,935      |
| Kochi   | Lang2022      | -3.03  | 11.41    | 9.91     | 21,208      |
| Kochi   | Potapov2021   | -0.55  | 7.06     | 1.99     | 21,203      |
| Hyogo   | Pauls2024     | -4.19  | 9.71     | 8.18     | 98,050      |
| Hyogo   | Tolan2024     | -0.50  | 5.22     | 0.34     | 97,842      |
| Hyogo   | Lang2022      | -7.10  | 12.11    | 11.31    | 97,960      |
| Hyogo   | Potapov2021   | -2.82  | 8.31     | 7.02     | 96,511      |
| Tochigi | GEDI          | -2.67  | 11.65    | -2.83    | 223         |
| Tochigi | Pauls2024     | -2.01  | 9.78     | 7.70     | 138,741     |
| Tochigi | Tolan2024     | -3.63  | 10.78    | -9.61    | 131,877     |
| Tochigi | Lang2022      | -2.34  | 9.89     | 8.10     | 133,409     |
| Tochigi | Potapov2021   | -0.39  | 6.06     | 2.42     | 135,026     |

**Note**: The GEDI sample size is significantly lower due to the sparse nature of the data, even after fixing the critical data loss bug during the alignment process. The negative R² values across the board highlight the need for a non-linear modeling approach.
