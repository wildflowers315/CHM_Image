import ee
import pytest
import json
import os
from l2a_gedi_source import get_gedi_vector_data, export_gedi_vector_points

def initialize_ee():
    """Initialize Earth Engine with project ID."""
    EE_PROJECT_ID = "my-project-423921"
    ee.Initialize(project=EE_PROJECT_ID)

def load_aoi(aoi_path: str) -> ee.Geometry:
    """Load AOI from GeoJSON file."""
    if not os.path.exists(aoi_path):
        raise FileNotFoundError(f"AOI file not found: {aoi_path}")
    
    with open(aoi_path, 'r') as f:
        geojson_data = json.load(f)
    
    def create_geometry(geom_type: str, coords: list) -> ee.Geometry:
        if geom_type == 'Polygon':
            return ee.Geometry.Polygon(coords)
        elif geom_type == 'MultiPolygon':
            return ee.Geometry.MultiPolygon(coords[0])
        else:
            raise ValueError(f"Unsupported geometry type: {geom_type}")
    
    if geojson_data['type'] == 'FeatureCollection':
        if not geojson_data['features']:
            raise ValueError("Empty FeatureCollection")
        geometry = geojson_data['features'][0]['geometry']
        return create_geometry(geometry['type'], geometry['coordinates'])
    elif geojson_data['type'] in ['Polygon', 'MultiPolygon']:
        return create_geometry(geojson_data['type'], geojson_data['coordinates'])
    else:
        raise ValueError(f"Unsupported GeoJSON type: {geojson_data['type']}")

def test_gedi_granule_discovery():
    """Test GEDI granule discovery for the AOI."""
    initialize_ee()
    
    # Load AOI
    aoi_path = "downloads/new_aoi.geojson"  # Updated path
    aoi = load_aoi(aoi_path)
    
    print(f"AOI bounds: {aoi.bounds().getInfo()}")
    
    # Load the granule index
    granule_index = ee.FeatureCollection('LARSE/GEDI/GEDI02_A_002_INDEX')
    print(f"Total granules in index: {granule_index.size().getInfo()}")
    
    # Filter granules that intersect your AOI
    granules = granule_index.filterBounds(aoi)
    print(f"Granules intersecting AOI: {granules.size().getInfo()}")
    
    # Get asset IDs of intersecting granules
    granule_list = granules.aggregate_array('asset_id').getInfo()
    print(f"Found {len(granule_list)} GEDI granules covering the AOI")
    
    if granule_list:
        print("First few granules:")
        for i, granule_id in enumerate(granule_list[:5]):
            print(f"  {i+1}: {granule_id}")
    
    return granule_list

def test_gedi_granule_loading():
    """Test loading individual GEDI granules."""
    initialize_ee()
    
    # Load AOI
    aoi_path = "downloads/new_aoi.geojson"  # Updated path
    aoi = load_aoi(aoi_path)
    
    # Get granules
    granule_index = ee.FeatureCollection('LARSE/GEDI/GEDI02_A_002_INDEX')
    granules = granule_index.filterBounds(aoi)
    granule_list = granules.aggregate_array('asset_id').getInfo()
    
    if not granule_list:
        print("No granules found, trying alternative approach...")
        # Try with a larger buffer
        aoi_buffered = aoi.buffer(5000)  # 5km buffer
        granules = granule_index.filterBounds(aoi_buffered)
        granule_list = granules.aggregate_array('asset_id').getInfo()
        print(f"Found {len(granule_list)} granules with buffer")
    
    if granule_list:
        # Test loading the first granule
        test_granule = granule_list[0]
        print(f"Testing granule: {test_granule}")
        
        try:
            gedi_fc = ee.FeatureCollection(test_granule)
            print(f"Successfully loaded granule with {gedi_fc.size().getInfo()} footprints")
            
            # Check available properties
            first_feature = gedi_fc.first()
            properties = first_feature.propertyNames().getInfo()
            print(f"Available properties: {properties}")
            
            # Filter by AOI
            gedi_aoi = gedi_fc.filterBounds(aoi)
            print(f"Footprints in AOI: {gedi_aoi.size().getInfo()}")
            
            return gedi_aoi
            
        except Exception as e:
            print(f"Failed to load granule: {e}")
            return None
    
    return None

def test_gedi_vector_data_function():
    """Test the complete get_gedi_vector_data function."""
    initialize_ee()
    
    # Load AOI
    aoi_path = "downloads/new_aoi.geojson"  # Updated path
    aoi = load_aoi(aoi_path)
    
    # Test parameters
    start_date = "2020-01-01"
    end_date = "2020-12-31"
    quantile = "rh98"  # Use rh98 to match your original data
    
    print("Testing get_gedi_vector_data function...")
    gedi_fc = get_gedi_vector_data(aoi, start_date, end_date, quantile)
    
    if gedi_fc is not None:
        print(f"Successfully loaded {gedi_fc.size().getInfo()} GEDI footprints")
        
        # Test export
        prefix = "test_gedi_vector"
        export_gedi_vector_points(gedi_fc, prefix)
    else:
        print("Failed to load GEDI vector data")
    
    return gedi_fc

def test_alternative_gedi_approach():
    """Test alternative approach using monthly data but as vector."""
    initialize_ee()
    
    # Load AOI
    aoi_path = "downloads/new_aoi.geojson"  # Updated path
    aoi = load_aoi(aoi_path)
    
    print("Testing alternative approach with monthly GEDI data...")
    
    # Try using the monthly data but convert to points
    gedi_monthly = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')
    
    # Filter by date and region
    gedi_filtered = gedi_monthly.filterDate("2020-01-01", "2020-12-31").filterBounds(aoi)
    
    print(f"Monthly GEDI images: {gedi_filtered.size().getInfo()}")
    
    if gedi_filtered.size().getInfo() > 0:
        # Convert to points
        gedi_points = gedi_filtered.select('rh98').mosaic().sample(
            region=aoi,
            scale=25,
            numPixels=1000,
            seed=42
        )
        
        print(f"Converted to {gedi_points.size().getInfo()} points")
        
        # Add coordinates more robustly
        def add_coordinates(feature):
            try:
                coords = feature.geometry().coordinates()
                lon = coords.get(0)
                lat = coords.get(1)
                rh = feature.get('rh98')
                return feature.set({
                    'longitude': lon,
                    'latitude': lat,
                    'rh': rh
                })
            except Exception as e:
                print(f"Error processing feature: {e}")
                return None
        
        gedi_points = gedi_points.map(add_coordinates)
        
        # Filter out null values
        gedi_points = gedi_points.filter(ee.Filter.notNull(['longitude', 'latitude', 'rh']))
        
        # Export
        prefix = "test_gedi_monthly_points"
        export_gedi_vector_points(gedi_points, prefix)
        
        return gedi_points
    
    return None

def test_granule_properties():
    """Test to check the properties of the granule index."""
    initialize_ee()
    
    # Load AOI
    aoi_path = "downloads/new_aoi.geojson"
    aoi = load_aoi(aoi_path)
    
    # Load the granule index
    granule_index = ee.FeatureCollection('LARSE/GEDI/GEDI02_A_002_INDEX')
    
    # Convert dates to proper format for filtering
    start_datetime = "2020-01-01T00:00:00Z"
    end_datetime = "2020-12-31T23:59:59Z"
    
    # Filter granules by date and location
    granules = granule_index.filter(
        f'time_start > "{start_datetime}" && time_end < "{end_datetime}"'
    ).filterBounds(aoi)
    
    print(f"Found {granules.size().getInfo()} granules in date range and AOI")
    
    # Get the first granule to check properties
    first_granule = granules.first().getInfo()
    print(f"First granule properties: {first_granule['properties']}")
    
    # Get all property names
    property_names = granules.first().propertyNames().getInfo()
    print(f"Available property names: {property_names}")
    
    return first_granule

if __name__ == "__main__":
    print("Running GEDI vector data tests...")
    
    print("\n0. Testing granule properties...")
    granule_info = test_granule_properties()
    
    print("\n1. Testing granule discovery...")
    granules = test_gedi_granule_discovery()
    
    print("\n2. Testing granule loading...")
    gedi_data = test_gedi_granule_loading()
    
    print("\n3. Testing complete function...")
    vector_data = test_gedi_vector_data_function()
    
    print("\n4. Testing alternative approach...")
    alt_data = test_alternative_gedi_approach()
    
    print("\nTests completed!") 