import unittest
import numpy as np
import torch
import tempfile
import os
from pathlib import Path
import rasterio
from rasterio.transform import from_origin

from .patch_loader import PatchDataset, create_patch_dataloader

class TestPatchLoader(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        # Create temporary directory for test patches
        self.temp_dir = tempfile.TemporaryDirectory()
        self.patch_dir = Path(self.temp_dir.name)
        
        # Create test patch data
        self.create_test_patches()
        
        # Create test GEDI data
        self.gedi_data = {
            'test_patch_0000': np.random.rand(256, 256),
            'test_patch_0001': np.random.rand(256, 256)
        }
    
    def tearDown(self):
        """Clean up test data."""
        self.temp_dir.cleanup()
    
    def create_test_patches(self):
        """Create test patch files."""
        # Define test bands
        band_names = [
            'S1_VV', 'S1_VH',
            'S2_B02', 'S2_B03', 'S2_B04', 'S2_B08',
            'ALOS2_HH', 'ALOS2_HV',
            'elevation', 'slope', 'aspect',
            'canopy_height', 'NDVI'
        ]
        
        # Create two test patches
        for i in range(2):
            # Create random data for each band
            data = np.random.rand(len(band_names), 256, 256)
            
            # Save as GeoTIFF
            patch_path = self.patch_dir / f'test_patch_{i:04d}.tif'
            with rasterio.open(
                patch_path,
                'w',
                driver='GTiff',
                height=256,
                width=256,
                count=len(band_names),
                dtype=data.dtype,
                crs='EPSG:4326',
                transform=from_origin(0, 0, 10, 10)
            ) as dst:
                dst.write(data)
                dst.descriptions = band_names
    
    def test_patch_dataset_initialization(self):
        """Test PatchDataset initialization."""
        dataset = PatchDataset(self.patch_dir)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(len(dataset.band_names), 13)
    
    def test_patch_dataset_getitem(self):
        """Test PatchDataset __getitem__ method."""
        dataset = PatchDataset(self.patch_dir, self.gedi_data)
        
        # Get first patch
        patch_data = dataset[0]
        
        # Check output structure
        self.assertIn('patch', patch_data)
        self.assertIn('mask', patch_data)
        self.assertIn('height', patch_data)
        self.assertIn('metadata', patch_data)
        
        # Check tensor shapes
        self.assertEqual(patch_data['patch'].shape, (13, 256, 256))
        self.assertEqual(patch_data['mask'].shape, (1, 256, 256))
        self.assertEqual(patch_data['height'].shape, (256, 256))
        
        # Check metadata
        self.assertIn('transform', patch_data['metadata'])
        self.assertIn('crs', patch_data['metadata'])
        self.assertIn('patch_id', patch_data['metadata'])
        self.assertIn('band_names', patch_data['metadata'])
    
    def test_patch_normalization(self):
        """Test patch normalization."""
        dataset = PatchDataset(self.patch_dir)
        patch_data = dataset[0]
        
        # Check that values are normalized
        normalized_patch = patch_data['patch']
        
        # S1 bands should be in [-1, 1]
        self.assertTrue(torch.all(normalized_patch[0:2] >= -1))
        self.assertTrue(torch.all(normalized_patch[0:2] <= 1))
        
        # S2 bands should be in [0, 1]
        self.assertTrue(torch.all(normalized_patch[2:6] >= 0))
        self.assertTrue(torch.all(normalized_patch[2:6] <= 1))
        
        # NDVI should be in [-1, 1]
        self.assertTrue(torch.all(normalized_patch[-1] >= -1))
        self.assertTrue(torch.all(normalized_patch[-1] <= 1))
    
    def test_patch_dataloader(self):
        """Test create_patch_dataloader function."""
        dataloader = create_patch_dataloader(
            patch_dir=self.patch_dir,
            gedi_data=self.gedi_data,
            batch_size=2
        )
        
        # Check dataloader
        self.assertEqual(len(dataloader), 1)  # 2 patches / batch_size=2
        
        # Get first batch
        batch = next(iter(dataloader))
        
        # Check batch structure
        self.assertIn('patch', batch)
        self.assertIn('mask', batch)
        self.assertIn('height', batch)
        self.assertIn('metadata', batch)
        
        # Check batch shapes
        self.assertEqual(batch['patch'].shape, (2, 13, 256, 256))
        self.assertEqual(batch['mask'].shape, (2, 1, 256, 256))
        self.assertEqual(batch['height'].shape, (2, 256, 256))

if __name__ == '__main__':
    unittest.main() 