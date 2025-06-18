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
        
        # 'local_beam_elevatio' < 1.5◦
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
    
    # Add diagnostic information
    print("Checking GEDI data values...")
    print(f"Number of GEDI footprints: {gedi_filtered.size().getInfo()}")
    
    # Convert GEDI points to a single image with one pixel per footprint
    print("\nConverting GEDI points to raster with one pixel per footprint...")
    print(f"Using quantile: {quantile}")
    
    # First mosaic the collection to get all valid measurements
    gedi_mosaic = gedi_filtered.select(quantile).mosaic()
    
    # Set the projection to match GEDI's native resolution (25m)
    gedi_rh = gedi_mosaic.reproject(
        crs='EPSG:4326',
        scale=25  # GEDI's native resolution
    ).rename("rh")
    
    # Add diagnostic information about the raster values
    print("\nChecking GEDI raster values...")
    stats = gedi_rh.reduceRegion(
        reducer=ee.Reducer.minMax().combine(
            reducer2=ee.Reducer.mean(),
            sharedInputs=True
        ).combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True
        ),
        geometry=aoi,
        scale=25,  # Use GEDI's native resolution
        maxPixels=1e13
    ).getInfo()
    
    if stats and 'rh_min' in stats and stats['rh_min'] is not None:
        print("GEDI height statistics:")
        print(f"Min height: {stats['rh_min']:.2f} meters")
        print(f"Max height: {stats['rh_max']:.2f} meters")
        print(f"Mean height: {stats['rh_mean']:.2f} meters")
        print(f"StdDev height: {stats['rh_stdDev']:.2f} meters")
    else:
        print("No valid height values found in the raster")
        raise ValueError("No valid height values found in the raster")
    
    # Count non-null pixels
    non_null = gedi_rh.unmask(0).gt(0).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=25,  # Use GEDI's native resolution
        maxPixels=1e13
    ).getInfo()
    
    print(f"Number of non-null pixels: {non_null.get('rh', 0)}")
    
    gedi_filtered = gedi_rh
    # Add debug information
    print("GEDI data processing complete")
    
    return gedi_filtered

def export_gedi_points(gedi_data: ee.Image, aoi: ee.Geometry, prefix: str):
    """
    Export GEDI points as CSV and GeoJSON with height values and coordinates.
    
    Args:
        gedi_data: GEDI data as Earth Engine Image
        aoi: Area of interest geometry
        prefix: Prefix for the export filename
    """
    # First check if we have any valid data
    valid_pixels = gedi_data.select('rh').gt(0).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=25,  # Use GEDI's native resolution
        maxPixels=1e13
    ).get('rh')
    
    valid_count = int(round(valid_pixels.getInfo()))
    print(f"Number of valid GEDI pixels in the area: {valid_count}")
    
    if valid_count == 0:
        print("No valid GEDI points found in the area. Skipping export.")
        return
    
    # Convert GEDI raster to points using a more robust approach
    # First, create a mask of valid pixels
    valid_mask = gedi_data.select('rh').gt(0)
    
    # Convert to points using reduceToImage and then sample
    gedi_points = valid_mask.reduceToImage(
        properties=['rh'],
        reducer=ee.Reducer.first()
    ).sample(
        region=aoi,
        scale=25,  # Use GEDI's native resolution
        numPixels=valid_count,
        seed=42
    )
    
    # Add coordinates to the properties and ensure we have valid points
    def add_coordinates(feature):
        coords = feature.geometry().coordinates()
        lon = coords.get(0)
        lat = coords.get(1)
        rh = feature.get('rh')
        return feature.set({
            'longitude': lon,
            'latitude': lat,
            'rh': rh
        })
    
    gedi_points = gedi_points.map(add_coordinates)
    
    # Filter out any points that might have null coordinates
    gedi_points = gedi_points.filter(ee.Filter.notNull(['longitude', 'latitude', 'rh']))
    
    # Get the number of points after filtering
    point_count = gedi_points.size().getInfo()
    print(f"Number of points after filtering: {point_count}")
    
    if point_count == 0:
        print("No valid points after filtering. Skipping export.")
        return
    
    # Export as CSV
    csv_export_task = ee.batch.Export.table.toDrive(
        collection=gedi_points,
        description=f"{prefix}_gedi_points_csv",
        fileNamePrefix=f"{prefix}_gedi_points",
        folder='GEE_exports',
        fileFormat='CSV',
        selectors=['longitude', 'latitude', 'rh']
    )
    
    # Export as GeoJSON
    geojson_export_task = ee.batch.Export.table.toDrive(
        collection=gedi_points,
        description=f"{prefix}_gedi_points_geojson",
        fileNamePrefix=f"{prefix}_gedi_points",
        folder='GEE_exports',
        fileFormat='GeoJSON',
        selectors=['longitude', 'latitude', 'rh']
    )
    
    # Start the exports
    csv_export_task.start()
    geojson_export_task.start()
    
    print(f"Started export of GEDI points:")
    print(f"- CSV task ID: {csv_export_task.id}")
    print(f"- GeoJSON task ID: {geojson_export_task.id}")
    print("The files will be available in your Google Drive once the exports complete.")

def get_gedi_vector_data(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    quantile: str
) -> ee.FeatureCollection:
    """
    Get GEDI L2A vector data for the specified area and time period.
    
    Args:
        aoi: Area of interest as Earth Engine Geometry
        start_date: Start date for GEDI data (YYYY-MM-DD)
        end_date: End date for GEDI data (YYYY-MM-DD)
        quantile: Quantile for GEDI data (e.g., 'rh100')
    
    Returns:
        ee.FeatureCollection: GEDI vector data points
    """
    # Load the granule index
    granule_index = ee.FeatureCollection('LARSE/GEDI/GEDI02_A_002_INDEX')
    
    # Convert dates to proper format for filtering
    start_datetime = f"{start_date}T00:00:00Z"
    end_datetime = f"{end_date}T23:59:59Z"
    
    # Filter granules by date and location
    granules = granule_index.filter(
        f'time_start > "{start_datetime}" && time_end < "{end_datetime}"'
    ).filterBounds(aoi)
    
    print(f"Found {granules.size().getInfo()} granules in date range and AOI")
    
    # Get the first few granules to test
    granules_list = granules.limit(10).getInfo()['features']
    
    if not granules_list:
        print("No GEDI granules found for the AOI and date range")
        return None
    
    print(f"Testing with {len(granules_list)} granules")
    
    # Load and merge all relevant granules
    gedi_collections = []
    for granule in granules_list:
        granule_id = granule['properties']['table_id']
        try:
            print(f"Loading granule: {granule_id}")
            gedi_fc = ee.FeatureCollection(granule_id)
            
            # Filter by AOI and apply quality filters
            gedi_filtered = gedi_fc.filterBounds(aoi)
            
            # Apply quality filters using ee.Filter instead of mapping
            gedi_filtered = gedi_filtered.filter(
                ee.Filter.gt(quantile, 0)
                .And(ee.Filter.eq("quality_flag", 1))
                .And(ee.Filter.eq("degrade_flag", 0))
                .And(ee.Filter.gt("sensitivity", 0.95))
                .And(ee.Filter.gt("beam", 4))
                .And(ee.Filter.lte("solar_elevation", 0))
                .And(ee.Filter.gt("num_detectedmodes", 0))
            )
            
            footprint_count = gedi_filtered.size().getInfo()
            print(f"  - Granule has {footprint_count} valid footprints after filtering")
            
            if footprint_count > 0:
                gedi_collections.append(gedi_filtered)
                
        except Exception as e:
            print(f"Failed to load granule {granule_id}: {e}")
            continue
    
    if not gedi_collections:
        print("No valid GEDI granules could be loaded")
        return None
    
    # Merge all collections
    gedi_merged = ee.FeatureCollection(gedi_collections).flatten()
    
    # Get the total number of footprints
    total_footprints = gedi_merged.size().getInfo()
    print(f"Total GEDI footprints after merging: {total_footprints}")
    
    if total_footprints == 0:
        print("No valid GEDI footprints found after quality filtering")
        return None
    
    return gedi_merged

def export_gedi_vector_points(gedi_fc: ee.FeatureCollection, prefix: str):
    """
    Export GEDI L2A vector points as CSV and GeoJSON with height values and coordinates.
    
    Args:
        gedi_fc: GEDI FeatureCollection
        prefix: Prefix for the export filename
    """
    if gedi_fc is None:
        print("No GEDI data to export")
        return
    
    # Get the number of points
    point_count = gedi_fc.size().getInfo()
    print(f"Number of GEDI points to export: {point_count}")
    
    if point_count == 0:
        print("No valid GEDI points found. Skipping export.")
        return
    
    # Add coordinates to the properties more robustly
    def add_coordinates(feature):
        try:
            coords = feature.geometry().coordinates()
            lon = coords.get(0)
            lat = coords.get(1)
            # Try different height properties
            rh = feature.get('rh100')
            if rh is None:
                rh = feature.get('rh98')
            if rh is None:
                rh = feature.get('rh95')
            if rh is None:
                rh = feature.get('rh90')
            
            return feature.set({
                'longitude': lon,
                'latitude': lat,
                'rh': rh
            })
        except Exception as e:
            print(f"Error processing feature: {e}")
            return None
    
    gedi_points = gedi_fc.map(add_coordinates)
    
    # Filter out any points that might have null coordinates
    gedi_points = gedi_points.filter(ee.Filter.notNull(['longitude', 'latitude', 'rh']))
    
    # Get the number of points after filtering
    final_count = gedi_points.size().getInfo()
    print(f"Number of points after filtering: {final_count}")
    
    if final_count == 0:
        print("No valid points after filtering. Skipping export.")
        return
    
    # Export as CSV
    csv_export_task = ee.batch.Export.table.toDrive(
        collection=gedi_points,
        description=f"{prefix}_gedi_points_csv",
        fileNamePrefix=f"{prefix}_gedi_points",
        folder='GEE_exports',
        fileFormat='CSV',
        selectors=['longitude', 'latitude', 'rh']
    )
    
    # Export as GeoJSON
    geojson_export_task = ee.batch.Export.table.toDrive(
        collection=gedi_points,
        description=f"{prefix}_gedi_points_geojson",
        fileNamePrefix=f"{prefix}_gedi_points",
        folder='GEE_exports',
        fileFormat='GeoJSON',
        selectors=['longitude', 'latitude', 'rh']
    )
    
    # Start the exports
    csv_export_task.start()
    geojson_export_task.start()
    
    print(f"Started export of GEDI points:")
    print(f"- CSV task ID: {csv_export_task.id}")
    print(f"- GeoJSON task ID: {geojson_export_task.id}")
    print("The files will be available in your Google Drive once the exports complete.")