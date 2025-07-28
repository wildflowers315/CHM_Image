#!/usr/bin/env python3
"""
Simple debug script to examine embedding file band structure
"""

import sys
import os
import glob
import subprocess

def check_with_gdalinfo(file_path):
    """Use gdalinfo to get band structure information"""
    try:
        result = subprocess.run(['gdalinfo', file_path], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            band_descriptions = []
            
            # Find band descriptions
            in_band_section = False
            current_band = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('Band '):
                    in_band_section = True
                    current_band = line
                elif in_band_section and line.startswith('Description = '):
                    desc = line.replace('Description = ', '')
                    band_descriptions.append((current_band, desc))
                elif in_band_section and (line.startswith('Band ') or line == ''):
                    in_band_section = False
                    
            return band_descriptions
    except Exception as e:
        print(f"Error running gdalinfo: {e}")
        return []

def main():
    print("=== SIMPLE BAND DEBUG ===\n")
    
    # Test files
    test_files = [
        "chm_outputs/dchm_04hf3_embedding_bandNum69_scale10_patch0044.tif",
        "chm_outputs/dchm_04hf3_embedding_bandNum70_scale10_patch0001.tif"
    ]
    
    print("1. CHECKING FILE PATTERN MATCHING:")
    pattern = "chm_outputs/*04hf3*embedding*.tif"
    matches = glob.glob(pattern)
    print(f"Pattern: {pattern}")
    print(f"Number of matches: {len(matches)}")
    print(f"Sample matches:")
    for match in matches[:5]:
        print(f"  - {match}")
    print()
    
    print("2. CHECKING SPECIFIC TEST FILES:")
    for file_path in test_files:
        exists = os.path.exists(file_path)
        print(f"  - {file_path}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
    print()
    
    # Find an existing file to analyze
    available_file = None
    for file_path in test_files:
        if os.path.exists(file_path):
            available_file = file_path
            break
    
    if not available_file and matches:
        available_file = matches[0]
    
    if available_file:
        print(f"3. ANALYZING FILE: {available_file}")
        print("Getting band structure with gdalinfo...")
        
        band_descriptions = check_with_gdalinfo(available_file)
        
        if band_descriptions:
            print("Band descriptions found:")
            global_height_bands = ['ch_pauls2024', 'ch_tolan2024', 'ch_lang2022', 'ch_potapov2021']
            
            for band_info, desc in band_descriptions:
                print(f"  {band_info}: {desc}")
                
            print("\n4. CHECKING FOR GLOBAL HEIGHT PRODUCTS:")
            found_products = []
            for target_band in global_height_bands:
                found = False
                for band_info, desc in band_descriptions:
                    if target_band in desc:
                        found_products.append((target_band, desc))
                        found = True
                        print(f"  ✅ {target_band}: FOUND - {desc}")
                        break
                if not found:
                    print(f"  ❌ {target_band}: NOT FOUND")
            
            print(f"\nSUMMARY: Found {len(found_products)} out of {len(global_height_bands)} global height products")
            
        else:
            print("❌ Could not extract band descriptions from gdalinfo")
    else:
        print("❌ No embedding files found to analyze!")

if __name__ == "__main__":
    main()