import ee


def get_dem_data(aoi: ee.Geometry):
    try:
        print("Downloading GMTED data")
        dem = ee.Image("USGS/GMTED2010_FULL").select(['min'], ['elevation']) #"USGS/GMTED2010" was deprecated.
        slope = ee.Terrain.slope(dem)
        aspect = ee.Terrain.aspect(dem)
        dem_data = dem.addBands(slope).addBands(aspect).select(['elevation', 'slope', 'aspect'], ['GMTED_elevation', 'GMTED_slope', 'GMTED_aspect']).clip(aoi)
    except Exception as e:
        print(f"Error loading GMTED data: {e}")
        dem_data = None
    try:
        # print(f"Downloading SRTM data instead{}")
        print("Donwloading SRTM data")
        dem = ee.Image("USGS/SRTMGL1_003").select('elevation')
        slope = ee.Terrain.slope(dem)
        aspect = ee.Terrain.aspect(dem)
        dem_data_SRTM = dem.addBands(slope).addBands(aspect).select(['elevation', 'slope', 'aspect'], ['SRTM_elevation', 'SRTM_slope', 'SRTM_aspect']).clip(aoi)
        dem_data = dem_data.addBands(dem_data_SRTM)
    except Exception as e:
        print(f"Error loading SRTM data: {e}")
    try:    
        print("Downloading ALOS AW3D30 data")
        # dem = ee.ImageCollection("JAXA/ALOS/AW3D30/V4_1").mosaic().select('DSM')
        dem = ee.ImageCollection("JAXA/ALOS/AW3D30/V3_2").mosaic().select('DSM')
        # proj = dem.select(0).projection()
        # slope = ee.Terrain.slope(dem.setDefaultProjection(proj)) # does not work
        # aspect = ee.Terrain.aspect(dem.setDefaultProjection(proj)) # does not work
        # slope = ee.Terrain.slope(dem)
        # aspect = ee.Terrain.aspect(dem)
        # dem_data_AW3D30 = dem.addBands(slope).addBands(aspect).select(['DSM', 'slope', 'aspect'], ['AW3D30_elevation', 'AW3D30_slope', 'AW3D30_aspect']).clip(aoi)
        dem_data_AW3D30 = dem.rename(['AW3D30_elevation'])
        dem_data = dem_data.addBands(dem_data_AW3D30)
    except Exception as e:
        print(f"Error loading ALOS AW3D30 data: {e}")
    
    try:
        print("Downloading GLO30 data")
        dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").mosaic().select('DEM')
        slope = ee.Terrain.slope(dem)
        aspect = ee.Terrain.aspect(dem)
        dem_data_GLO30 = dem.addBands(slope).addBands(aspect).select(['DEM', 'slope', 'aspect'], ['GLO30_elevation', 'GLO30_slope', 'GLO30_aspect']).clip(aoi)
        dem_data = dem_data.addBands(dem_data_GLO30)
    except Exception as e:
        print(f"Error loading GLO30 data: {e}")
        
    return dem_data
