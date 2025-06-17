import unittest
import torch
import numpy as np
import torch.nn.functional as F

from models.unet_3d import Height3DUNet, create_3d_unet

class Test3DUNet(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.batch_size = 2
        self.in_channels = 13  # Number of input bands
        self.height = 256
        self.width = 256
        self.depth = 12  # Temporal dimension (months)
        
        # Create test input
        self.x = torch.randn(
            self.batch_size,
            self.in_channels,
            self.depth,
            self.height,
            self.width
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = Height3DUNet(
            in_channels=self.in_channels,
            n_classes=1,
            base_channels=64
        )
        
        # Check model structure
        self.assertEqual(model.in_channels, self.in_channels)
        self.assertEqual(model.n_classes, 1)
        self.assertEqual(model.base_channels, 64)
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        model = Height3DUNet(
            in_channels=self.in_channels,
            n_classes=1,
            base_channels=64
        )
        
        # Forward pass
        output = model(self.x)
        
        # Check output shape
        expected_shape = (self.batch_size, 1, self.height, self.width)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output values
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_forward_pass_without_temporal(self):
        """Test forward pass with input without temporal dimension."""
        model = Height3DUNet(
            in_channels=self.in_channels,
            n_classes=1,
            base_channels=64
        )
        
        # Create input without temporal dimension
        x_no_temp = torch.randn(
            self.batch_size,
            self.in_channels,
            self.height,
            self.width
        )
        
        # Forward pass
        output = model(x_no_temp)
        
        # Check output shape
        expected_shape = (self.batch_size, 1, self.height, self.width)
        self.assertEqual(output.shape, expected_shape)
    
    def test_model_creation(self):
        """Test model creation function."""
        model = create_3d_unet(
            in_channels=self.in_channels,
            n_classes=1,
            base_channels=64
        )
        
        # Check model structure
        self.assertEqual(model.in_channels, self.in_channels)
        self.assertEqual(model.n_classes, 1)
        self.assertEqual(model.base_channels, 64)
    
    def test_weight_initialization(self):
        """Test weight initialization."""
        model = Height3DUNet(
            in_channels=self.in_channels,
            n_classes=1,
            base_channels=64
        )
        
        # Check that weights are initialized
        for name, param in model.named_parameters():
            if 'weight' in name:
                self.assertFalse(torch.all(param == 0))
                self.assertTrue(torch.all(torch.isfinite(param)))
    
    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        model = Height3DUNet(
            in_channels=self.in_channels,
            n_classes=1,
            base_channels=64
        )
        
        # Forward pass
        output = model(self.x)
        
        # Create dummy target
        target = torch.randn_like(output)
        
        # Compute loss
        loss = F.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.all(torch.isfinite(param.grad)))

if __name__ == '__main__':
    unittest.main() 