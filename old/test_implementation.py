import os
import torch
import numpy as np
from pathlib import Path
import ee
import geemap
import rasterio
from rasterio.transform import from_origin
import json

from chm_main import load_aoi, create_patches_from_ee_image, initialize_ee, create_patch_geometries
from data.patch_loader import PatchDataset, create_patch_dataloader
from models.unet_3d import create_3d_unet

def test_patch_geometries():
    print("\nTesting create_patch_geometries...")
    initialize_ee()
    aoi_path = "downloads/dchm_09gd4.geojson"
    aoi = load_aoi(aoi_path)
    patch_size = 2560
    overlap = 0.1
    patches = create_patch_geometries(aoi, patch_size, overlap)
    print(f"Number of patches created: {len(patches)}")
    if patches:
        patch_info = patches[0].getInfo()
        print(f"First patch geometry: {patch_info}")
    assert len(patches) >= 1, "No patches were created!"

def test_implementation():
    """Test the current implementation with a real AOI."""
    print("Testing implementation with dchm_09gd4.geojson...")
    
    # Initialize Earth Engine
    try:
        initialize_ee()
        print("Earth Engine initialized successfully")
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        return
    
    # Load AOI
    aoi_path = "downloads/dchm_09gd4.geojson"
    try:
        aoi = load_aoi(aoi_path)
        print("AOI loaded successfully")
    except Exception as e:
        print(f"Error loading AOI: {e}")
        return
    
    # Create test data directory
    test_dir = Path("test_outputs")
    test_dir.mkdir(exist_ok=True)
    
    # Test patch creation
    try:
        print("\nTesting patch creation...")
        # Create a simple test image with multiple bands
        test_image = ee.Image.constant(1).rename('test_band')
        
        # Create patches
        patches = create_patches_from_ee_image(
            image=test_image,
            aoi=aoi,
            patch_size=2560,  # 2.56km patches
            scale=10  # 10m resolution
        )
        print(f"Created {len(patches)} patches")
        
        # Save patch information
        patch_info = []
        for i, patch in enumerate(patches):
            patch_info.append({
                'id': i,
                'bounds': patch['bounds'],
                'is_extruded': patch['is_extruded']
            })
        
        # Save patch information to file
        with open(test_dir / 'patch_info.json', 'w') as f:
            json.dump(patch_info, f, indent=2)
        print("Saved patch information to test_outputs/patch_info.json")
        
    except Exception as e:
        print(f"Error in patch creation: {e}")
        return
    
    # Test data loading pipeline
    try:
        print("\nTesting data loading pipeline...")
        # Create a test patch file
        test_patch_path = test_dir / 'test_patch_0000.tif'
        
        # Create test data
        band_names = [
            'S1_VV', 'S1_VH',
            'S2_B02', 'S2_B03', 'S2_B04', 'S2_B08',
            'ALOS2_HH', 'ALOS2_HV',
            'elevation', 'slope', 'aspect',
            'canopy_height', 'NDVI'
        ]
        
        # Create test data with proper dimensions for 3D patches
        # [bands, time_steps, height, width]
        data = np.random.rand(len(band_names), 12, 256, 256)
        
        with rasterio.open(
            test_patch_path,
            'w',
            driver='GTiff',
            height=256,
            width=256,
            count=len(band_names) * 12,  # Total bands including time steps
            dtype=data.dtype,
            crs='EPSG:4326',
            transform=from_origin(0, 0, 10, 10)
        ) as dst:
            # Write each band and time step
            for b in range(len(band_names)):
                for t in range(12):
                    band_idx = b * 12 + t + 1  # 1-based band index
                    dst.write(data[b, t], band_idx)
                    dst.set_band_description(band_idx, f"{band_names[b]}_t{t}")
        
        # Create test GEDI data
        gedi_data = {
            'test_patch_0000': np.random.rand(256, 256)
        }
        
        # Test dataset
        dataset = PatchDataset(test_dir, gedi_data)
        print(f"Dataset created with {len(dataset)} patches")
        
        if len(dataset) > 0:
            # Test dataloader
            dataloader = create_patch_dataloader(
                patch_dir=test_dir,
                gedi_data=gedi_data,
                batch_size=1
            )
            print("Dataloader created successfully")
            
            # Test loading a batch
            batch = next(iter(dataloader))
            print("Batch loaded successfully")
            print(f"Batch shapes:")
            print(f"- Patch: {batch['patch'].shape}")
            print(f"- Mask: {batch['mask'].shape}")
            print(f"- Height: {batch['height'].shape}")
        else:
            print("No patches with GEDI data available, skipping dataloader test")
        
    except Exception as e:
        print(f"Error in data loading pipeline: {e}")
        return
    
    # Test 3D U-Net model
    try:
        print("\nTesting 3D U-Net model...")
        # Create model
        model = create_3d_unet(
            in_channels=13,  # Number of input bands
            n_classes=1,
            base_channels=64
        )
        print("Model created successfully")
        
        # Test forward pass
        x = torch.randn(1, 13, 12, 256, 256)  # [batch, channels, time, height, width]
        output = model(x)
        print(f"Forward pass successful")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
    except Exception as e:
        print(f"Error in model testing: {e}")
        return
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_patch_geometries()
    test_implementation() 