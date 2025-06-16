import ee
from typing import Union, List

def get_gedi_data(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    quantile: str
) -> ee.ImageCollection:
    """
    Get GEDI L2A data for the specified area and time period.
    
    Args:
        aoi: Area of interest as Earth Engine Geometry
        start_date: Start date for GEDI data
        end_date: End date for GEDI data
        quantile: Quantile for GEDI data (e.g., 'rh100')
    
    Returns:
        ee.ImageCollection: GEDI data points
    """
    # Import GEDI L2A dataset
    gedi = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')
    
    # Filter by date and region
    gedi_filtered = gedi.filterDate(start_date, end_date) \
                        .filterBounds(aoi)

    # Select quality metrics and height metrics
    # gedi_filtered = gedi_filtered.select([
    #     'quality_flag',
    #     'degrade_flag',
    #     'sensitivity',
    #     'solar_elevation',
    #     'rh100',
    #     'rh98',
    #     'rh95',
    #     'rh90',
    #     'rh75',
    #     'rh50',
    #     'rh25',
    #     'rh10'
    # ])
    
    # https://lpdaac.usgs.gov/documents/982/gedi_l2a_dictionary_P003_v2.html
    def qualityMask(img):
        # First check if we have valid data
        has_data = img.select(quantile).gt(0)
        # Then apply quality filters
        quality_ok = img.select("quality_flag").eq(1) # Only select good quality data
        degrade_ok = img.select("degrade_flag").eq(0) # Only select no degrade data
        sensitivity_ok = img.select('sensitivity').gt(0.95) # Only select high sensitivity data 0.90 is the minimum
        fullPowerBeam_ok = img.select('beam').gt(4) # Only select full power beam with BEAM0101, BEAM0110, BEAM1000, BEAM1011
        solar_elevation_ok = img.select('solar_elevation').lte(0) # less tan 0 lead to day time distorsion by sun light
        detect_node_ok = img.select('num_detectedmodes').gt(0) # Only select detect node data
        # Only select data with elevation difference less than 50m
        elev_diff_ok = img.select('elev_lowestmode').subtract(img.select('digital_elevation_model_srtm')).lt(50) \
            # .And(img.select('elev_lowestmode').subtract(img.select('digital_elevation_model_srtm')).lt(-50))
        
        # ‘local_beam_elevatio’ < 1.5◦
        # local_beam_elevation_ok = img.select('local_beam_elevation').lt(1.5)

        # Combine all conditions
        return img.updateMask(has_data) \
                 .updateMask(quality_ok) \
                 .updateMask(degrade_ok) \
                .updateMask(sensitivity_ok) \
                .updateMask(fullPowerBeam_ok) \
                .updateMask(solar_elevation_ok) \
                .updateMask(detect_node_ok) \
                .updateMask(elev_diff_ok) \
                    
                # .updateMask(local_beam_elevation_ok) \
                    
    
    # Select and rename the quantile
    # def rename_property(image):
    #     return image.select([quantile]).rename('rh')
    
    # gedi_filtered = gedi_filtered.map(rename_property)
    
    # Then apply quality mask
    gedi_filtered = gedi_filtered.map(qualityMask)
    gedi_filtered_rh = gedi_filtered.select(quantile).mosaic().rename("rh")
    gedi_additional_bands = [
        'digital_elevation_model',
        'digital_elevation_model_srtm',
        'elev_lowestmode',
    ]
    gedi_filtered = gedi_filtered.select(gedi_additional_bands).mosaic().addBands(gedi_filtered_rh)
    # Get all valid points by using reduce(ee.Reducer.toCollection())
    # Specify the property names we want to keep
    # gedi_points = gedi_filtered.select([quantile, 'quality_flag', 'degrade_flag']).reduce(
    #     ee.Reducer.toCollection([quantile])
    #     # ee.Reducer.toCollection(['quality_flag', 'degrade_flag', quantile])
    # )
    
    # Rename the quantile band to 'rh'
    # gedi_points = gedi_points.rename(quantile, "rh")
    
    return gedi_filtered