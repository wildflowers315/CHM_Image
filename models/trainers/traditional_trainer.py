import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Tuple, Optional, Dict

from dl_models import MLPRegressionModel, create_normalized_dataloader
from evaluate_predictions import calculate_metrics

def train_model(X: np.ndarray, y: np.ndarray, model_type: str = 'rf', 
                batch_size: int = 64, test_size: float = 0.2, feature_names: Optional[list] = None,
                n_bands: Optional[int] = None) -> Tuple[object, dict, dict]:
    """
    Training function for traditional models (RF/MLP only).
    
    Args:
        X: Feature matrix (extracted from patch for GEDI pixels)
        y: Target variable (extracted from patch for GEDI pixels)
        model_type: Type of model ('rf' or 'mlp')
        batch_size: Batch size for MLP training
        test_size: Proportion of data to use for validation
        feature_names: Optional list of feature names
        
    Returns:
        Trained model, training metrics, and feature importance/weights
    """
    if model_type not in ['rf', 'mlp']:
        raise ValueError(f"train_model only supports 'rf' and 'mlp', got '{model_type}'")
    
    if X.shape[0] < 10:  # Need minimum samples for train/val split
        raise ValueError(f"Insufficient GEDI pixels ({X.shape[0]}) for {model_type.upper()} training. Need at least 10.")
    
    print(f"Training {model_type.upper()} on {X.shape[0]} GEDI pixels with {X.shape[1]} features")
    
    # Split data for traditional models
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    if model_type == 'rf':
        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=5,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_val)
        train_metrics = calculate_metrics(y_pred, y_val)
        
        # Get feature importance
        importance = model.feature_importances_
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        importance_data = {
            name: float(imp) for name, imp in zip(feature_names, importance)
        }
        
    else:  # MLP model
        # Create normalized dataloaders
        train_loader, val_loader, scaler_mean, scaler_std = create_normalized_dataloader(
            X_train, X_val, y_train, y_val, batch_size=batch_size, n_bands=n_bands
        )
        
        # Initialize model
        model = MLPRegressionModel(input_size=X.shape[1])
        if torch.cuda.is_available():
            model = model.cuda()
            
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())
        num_epochs = 100
        best_val_loss = float('inf')
        
        # Training loop with tqdm progress bar
        for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
            model.train()
            for batch_X, batch_y in train_loader:
                if torch.cuda.is_available():
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_predictions = []
            val_targets = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    if torch.cuda.is_available():
                        batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                    outputs = model(batch_X)
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            val_metrics = calculate_metrics(val_predictions, val_targets)
            val_loss = val_metrics['RMSE']
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                train_metrics = val_metrics
        
        # Get feature importance (using weights of first layer as proxy)
        with torch.no_grad():
            weights = model.layers[0].weight.abs().mean(dim=0).cpu().numpy()
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(weights))]
            importance_data = {}
            for name, weight in zip(feature_names, weights):
                # Convert numpy.float32 to Python float
                weight_value = weight.item() if hasattr(weight, 'item') else float(weight)
                importance_data[name] = weight_value
        
        # Store normalization parameters with model
        model.scaler_mean = scaler_mean
        model.scaler_std = scaler_std
    
    # Sort importance by value
    importance_data = dict(sorted(importance_data.items(), key=lambda x: x[1], reverse=True))
    
    # Print metrics and top features
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.3f}")
    
    print("\nTop 5 Important Features:")
    for name, imp in list(importance_data.items())[:5]:
        print(f"{name}: {imp:.3f}")
    
    return model, train_metrics, importance_data
