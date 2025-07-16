# GEDI Patch Analysis Results

## Summary Statistics

**Total Embedding Patches**: 189 patches across all regions
**Patches with GEDI Data**: 91 patches (48.1%)
**Patches without GEDI Data**: 98 patches (51.9%)

## Regional Distribution

### Hyogo Region (dchm_04hf3)
- **Total patches**: 63
- **With GEDI**: 3 patches (4.8%)
- **Without GEDI**: 60 patches (95.2%)
- **GEDI patch numbers**: [0, 1, 7]

### Kochi Region (dchm_05LE4)
- **Total patches**: 63
- **With GEDI**: 63 patches (100%)
- **Without GEDI**: 0 patches (0%)
- **GEDI patch numbers**: [0-62] (all patches)

### Tochigi Region (dchm_09gd4)
- **Total patches**: 63
- **With GEDI**: 25 patches (39.7%)
- **Without GEDI**: 38 patches (60.3%)
- **GEDI patch numbers**: [0, 1, 2, 5, 6, 7, 11, 12, 13, 17, 18, 19, 20, 23, 24, 25, 26, 35, 36, 42, 43, 49, 50, 56, 62]

## Technical Details

### Band Structure
- **Patches with GEDI**: 70 bands total
  - Bands 1-64: Google Embedding v1 (64 bands)
  - Bands 65-69: Reference height data (ch_pauls2024, ch_tolan2024, ch_lang2022, ch_potapov2021, forest_mask)
  - Band 70: **GEDI data (rh)** - GEDI relative height measurements

- **Patches without GEDI**: 69 bands total
  - Bands 1-64: Google Embedding v1 (64 bands)
  - Bands 65-69: Reference height data (same as above)
  - No GEDI band

### Data Specifications
- **Dimensions**: 256×256 pixels per patch
- **Resolution**: 10m per pixel (2.56km × 2.56km coverage)
- **Data Type**: Float32
- **Coordinate System**: EPSG:4326
- **Value Range**: Pre-normalized to [-1, 1]

## Implementation Lists

### Python Lists for Code
```python
# Patch numbers containing GEDI data by region
DCHM_04HF3_GEDI_PATCHES = [0, 1, 7]
DCHM_05LE4_GEDI_PATCHES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
DCHM_09GD4_GEDI_PATCHES = [0, 1, 2, 5, 6, 7, 11, 12, 13, 17, 18, 19, 20, 23, 24, 25, 26, 35, 36, 42, 43, 49, 50, 56, 62]

# Check if patch has GEDI data
def has_gedi_data(area_id, patch_number):
    """Check if a patch contains GEDI data based on region and patch number"""
    if area_id == 'dchm_04hf3':
        return patch_number in DCHM_04HF3_GEDI_PATCHES
    elif area_id == 'dchm_05LE4':
        return patch_number in DCHM_05LE4_GEDI_PATCHES
    elif area_id == 'dchm_09gd4':
        return patch_number in DCHM_09GD4_GEDI_PATCHES
    return False

# Get expected band count
def get_expected_bands(area_id, patch_number):
    """Get expected number of bands for a patch"""
    return 70 if has_gedi_data(area_id, patch_number) else 69
```

## Key Findings

1. **Regional Variation**: GEDI coverage varies significantly by region:
   - Kochi has complete GEDI coverage (100%)
   - Tochigi has moderate coverage (39.7%)
   - Hyogo has minimal coverage (4.8%)

2. **Band Structure**: GEDI data is consistently stored in band 70 when present, making it easy to identify and extract.

3. **Data Quality**: All patches with GEDI data have 70 bands, while patches without GEDI have 69 bands, providing a reliable indicator.

4. **Implementation Impact**: For patch selection logic, you can use the patch number lists above to determine which patches have GEDI data available for training scenarios 2A, 2B, and 3.

## Usage for Training Scenarios

- **Scenario 1 (Reference-only)**: Can use all 189 patches
- **Scenario 2A/2B (Reference + GEDI)**: Must use only the 91 patches with GEDI data
- **Scenario 3 (GEDI adaptation)**: Focus on Tochigi region's 25 GEDI patches for fine-tuning

## File Location
Analysis conducted on patches in: `/home/WUR/ishik001/CHM_Image/chm_outputs/`
Pattern: `*embedding*scale10*.tif`