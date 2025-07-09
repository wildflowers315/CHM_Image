#!/usr/bin/env python3
"""
Ensemble MLP for combining GEDI shift-aware model and production MLP predictions
Scenario 2: Reference + GEDI Training (No Target Adaptation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EnsembleMLP(nn.Module):
    """
    MLP ensemble that combines predictions from:
    1. GEDI shift-aware U-Net model (spatial context)
    2. Production MLP model (pixel-level precision)
    
    Architecture:
    - Input: 2 features (GEDI prediction + MLP prediction)
    - Target: Reference height TIF supervision
    - Advantage: Combines spatial context with pixel-level accuracy
    """
    
    def __init__(self, input_dim=2, hidden_dims=[64, 32, 16], 
                 dropout_rate=0.3, use_residuals=False):
        super().__init__()
        
        self.use_residuals = use_residuals
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            
            # Dropout
            dropout_p = dropout_rate * (1 - i / len(hidden_dims))  # Decreasing dropout
            self.dropouts.append(nn.Dropout(dropout_p))
            
            prev_dim = hidden_dim
        
        # Output layer
        self.output = nn.Linear(prev_dim, 1)
        
        # Prediction confidence weighting (learnable)
        self.prediction_weights = nn.Parameter(torch.tensor([0.5, 0.5]))  # Equal initial weights
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, gedi_pred, mlp_pred):
        """
        Forward pass
        
        Args:
            gedi_pred: GEDI model predictions (batch_size,)
            mlp_pred: MLP model predictions (batch_size,)
            
        Returns:
            ensemble_pred: Combined predictions (batch_size,)
        """
        # Stack predictions as input features
        x = torch.stack([gedi_pred, mlp_pred], dim=1)  # (batch_size, 2)
        
        # Apply learned prediction weights
        weights = F.softmax(self.prediction_weights, dim=0)
        weighted_input = x * weights.unsqueeze(0)  # Broadcast weights
        
        # Forward through MLP layers
        for i, (layer, norm, dropout) in enumerate(zip(self.layers, self.norms, self.dropouts)):
            residual = x if self.use_residuals and layer.in_features == layer.out_features else None
            
            x = layer(weighted_input if i == 0 else x)
            x = norm(x)
            x = F.relu(x)
            x = dropout(x)
            
            # Add residual connection if applicable
            if residual is not None:
                x = x + residual
        
        # Output prediction
        output = self.output(x).squeeze(-1)
        
        return output
    
    def get_prediction_weights(self):
        """Get the learned prediction weights"""
        with torch.no_grad():
            weights = F.softmax(self.prediction_weights, dim=0)
            return {
                'gedi_weight': weights[0].item(),
                'mlp_weight': weights[1].item()
            }


class SimpleEnsembleMLP(nn.Module):
    """
    Simplified ensemble that learns to combine two predictions
    """
    
    def __init__(self):
        super().__init__()
        
        # Simple 2-input network
        self.ensemble = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )
        
        # Learnable weights for input predictions
        self.input_weights = nn.Parameter(torch.tensor([1.0, 1.0]))
    
    def forward(self, gedi_pred, mlp_pred):
        """Combine two predictions"""
        # Apply learnable input scaling
        scaled_gedi = gedi_pred * self.input_weights[0]
        scaled_mlp = mlp_pred * self.input_weights[1]
        
        # Stack and process
        x = torch.stack([scaled_gedi, scaled_mlp], dim=1)
        return self.ensemble(x).squeeze(-1)
    
    def get_weights(self):
        """Get current input weights"""
        return {
            'gedi_weight': self.input_weights[0].item(),
            'mlp_weight': self.input_weights[1].item()
        }


class AdaptiveEnsemble(nn.Module):
    """
    Adaptive ensemble that can adjust mixing based on prediction confidence
    """
    
    def __init__(self, use_confidence=True):
        super().__init__()
        
        self.use_confidence = use_confidence
        
        # Main ensemble network - always use 2 for now (predictions only)
        input_dim = 2  # GEDI + MLP predictions
        self.ensemble = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32), 
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Attention mechanism for dynamic weighting
        self.attention = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, gedi_pred, mlp_pred, gedi_conf=None, mlp_conf=None):
        """
        Forward with optional confidence scores
        
        Args:
            gedi_pred: GEDI predictions
            mlp_pred: MLP predictions  
            gedi_conf: GEDI prediction confidence (optional)
            mlp_conf: MLP prediction confidence (optional)
        """
        # Stack predictions
        predictions = torch.stack([gedi_pred, mlp_pred], dim=1)
        
        # Calculate attention weights
        attention_weights = self.attention(predictions)
        weighted_preds = predictions * attention_weights
        
        # Prepare ensemble input
        if self.use_confidence and gedi_conf is not None and mlp_conf is not None:
            confidences = torch.stack([gedi_conf, mlp_conf], dim=1)
            ensemble_input = torch.cat([weighted_preds, confidences], dim=1)
        else:
            # Use only weighted predictions (2D input)
            ensemble_input = weighted_preds.view(weighted_preds.size(0), -1)
        
        # Final ensemble prediction
        return self.ensemble(ensemble_input).squeeze(-1)


def create_ensemble_model(model_type='simple', **kwargs):
    """
    Factory function to create ensemble models
    
    Args:
        model_type: 'simple', 'advanced', or 'adaptive'
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Ensemble model instance
    """
    
    if model_type == 'simple':
        return SimpleEnsembleMLP()
    elif model_type == 'advanced':
        return EnsembleMLP(**kwargs)
    elif model_type == 'adaptive':
        return AdaptiveEnsemble(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test ensemble models
    batch_size = 100
    
    # Create sample predictions
    gedi_pred = torch.randn(batch_size) * 20 + 15  # Heights around 15m
    mlp_pred = torch.randn(batch_size) * 20 + 15   # Heights around 15m
    
    print("Testing Ensemble Models:")
    print("=" * 40)
    
    # Test Simple Ensemble
    simple_model = create_ensemble_model('simple')
    simple_out = simple_model(gedi_pred, mlp_pred)
    print(f"Simple Ensemble:")
    print(f"  Output shape: {simple_out.shape}")
    print(f"  Output range: {simple_out.min():.2f} - {simple_out.max():.2f}")
    print(f"  Weights: {simple_model.get_weights()}")
    
    # Test Advanced Ensemble  
    advanced_model = create_ensemble_model('advanced')
    advanced_out = advanced_model(gedi_pred, mlp_pred)
    print(f"\nAdvanced Ensemble:")
    print(f"  Output shape: {advanced_out.shape}")
    print(f"  Output range: {advanced_out.min():.2f} - {advanced_out.max():.2f}")
    print(f"  Weights: {advanced_model.get_prediction_weights()}")
    
    # Test Adaptive Ensemble
    adaptive_model = create_ensemble_model('adaptive')
    adaptive_out = adaptive_model(gedi_pred, mlp_pred)
    print(f"\nAdaptive Ensemble:")
    print(f"  Output shape: {adaptive_out.shape}")
    print(f"  Output range: {adaptive_out.min():.2f} - {adaptive_out.max():.2f}")
    
    print("\n✅ All ensemble models working correctly!")