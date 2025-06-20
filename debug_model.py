#!/usr/bin/env python3
"""Debug model checkpoint to understand the structure."""

import torch
from pathlib import Path

def debug_checkpoint(checkpoint_path):
    """Debug the checkpoint structure."""
    print(f"🔍 Debugging checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Checkpoint loaded successfully")
        
        print(f"\n📋 Checkpoint keys:")
        for key in checkpoint.keys():
            print(f"   - {key}: {type(checkpoint[key])}")
        
        # Check if model_state_dict exists
        if 'model_state_dict' in checkpoint:
            print(f"\n🔍 model_state_dict keys (first 10):")
            state_dict = checkpoint['model_state_dict']
            for i, key in enumerate(list(state_dict.keys())[:10]):
                tensor = state_dict[key]
                print(f"   {i+1}. {key}: {tensor.shape}")
                
            # Look for encoder1.0.weight specifically
            if 'encoder1.0.weight' in state_dict:
                weight = state_dict['encoder1.0.weight']
                print(f"\n🎯 Found encoder1.0.weight:")
                print(f"   Shape: {weight.shape}")
                print(f"   Input channels: {weight.shape[1]}")
                print(f"   Output channels: {weight.shape[0]}")
                
            # Look for final conv layer
            final_keys = [k for k in state_dict.keys() if 'final' in k.lower()]
            print(f"\n🎯 Final layer keys: {final_keys}")
            for key in final_keys:
                if 'weight' in key:
                    print(f"   {key}: {state_dict[key].shape}")
                    
        else:
            print(f"\n🔍 Direct checkpoint keys (first 10):")
            for i, key in enumerate(list(checkpoint.keys())[:10]):
                if hasattr(checkpoint[key], 'shape'):
                    print(f"   {i+1}. {key}: {checkpoint[key].shape}")
                else:
                    print(f"   {i+1}. {key}: {type(checkpoint[key])}")
                    
            # Look for encoder1.0.weight specifically
            if 'encoder1.0.weight' in checkpoint:
                weight = checkpoint['encoder1.0.weight']
                print(f"\n🎯 Found encoder1.0.weight:")
                print(f"   Shape: {weight.shape}")
                print(f"   Input channels: {weight.shape[1]}")
                print(f"   Output channels: {weight.shape[0]}")
    
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")

if __name__ == "__main__":
    checkpoint_path = "chm_outputs/2d_unet/final_model.pth"
    debug_checkpoint(checkpoint_path)