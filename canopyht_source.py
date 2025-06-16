import ee

def get_canopyht_data(
    aoi: ee.Geometry,
    # year: int,
    # start_date: str,
    # end_date: str
) -> ee.Image:
    
    # https://github.com/AI4Forest/Global-Canopy-Height-Map/tree/main
    ch_pauls2024 = ee.ImageCollection('projects/worldwidemap/assets/canopyheight2020') \
        .filterBounds(aoi).mosaic().select([0],['ch_pauls2024']).divide(100)  # Convert cm to m
    
    # https://gee-community-catalog.org/projects/meta_trees/#dataset-citation
    ch_tolan2024 = ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight") \
        .filterBounds(aoi).mosaic().select([0],['ch_tolan2024'])
    # treenotree = canopy_ht_2023.updateMask(canopy_ht_2023.gte(0))
    
    # https://gee-community-catalog.org/projects/canopy/#earth-engine-snippet
    ch_lang2022 = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1") \
        .select([0],['ch_lang2022'])
    standard_deviation_2022 = ee.Image("users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1") \
        .select([0],['ch_lang2022_stddev'])
    
    # https://www.sciencedirect.com/science/article/pii/S0034425720305381
    ch_potapov2021 = ee.ImageCollection("users/potapovpeter/GEDI_V27") \
        .filterBounds(aoi).mosaic().select([0],['ch_potapov2021']) # 30m resolution
    
    # Merge the images
    canopy_ht = ch_pauls2024.addBands(ch_tolan2024) \
        .addBands(ch_lang2022) \
        .addBands(ch_potapov2021) 
        # .addBands(standard_deviation_2022) \

    return canopy_ht.clip(aoi)