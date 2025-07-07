# Reference Height Training Enhancement Plan

## Executive Summary

This document outlines the plan to enhance the CHM modeling system by incorporating reference height TIF data from airborne LiDAR (Hyogo region) as an additional training supervision source, complementing the existing sparse GEDI supervision.

## Current State Analysis

### Dataset Overview
- **09gd4 (Tochigi)**: 60 patches (33 bandNum30 + 27 bandNum31) - Previsou training region but now validation region 
- **05LE4 (Hyogo)**: 63 patches (all bandNum31) - **TARGET: Dense reference height training**
- **04hf3 (Kochi)**: 36 patches (33 bandNum30 + 3 bandNum31) - Secondary validation region

### Current Training Supervision
- **GEDI L2A**: Sparse supervision (<0.3% pixel coverage)
- **Reference TIF**: Dense validation data (100% coverage) - **UNDERUTILIZED FOR TRAINING**

## Implementation Plan

### Experimental Design: Three-Scenario Comparison Framework

This plan implements a comprehensive experimental framework to compare three distinct training strategies for canopy height modeling. The goal is to determine the optimal approach for leveraging reference height data while understanding the contribution of GEDI supervision and cross-region adaptation.

## Three-Scenario Comparison Framework

### Scenario 1: Reference-Only Training (No GEDI)
**Objective**: Evaluate performance of dense reference height supervision without GEDI data
- **Training Data**: Hyogo patches with reference height TIF supervision only
- **Model**: 2D U-Net trained exclusively on reference height data
- **Validation**: Apply trained model directly to Tochigi and Kochi regions
- **Key Question**: How well does reference-only training generalize across regions?

### Scenario 2: Reference + GEDI Training (No Target Adaptation)
**Objective**: Evaluate combined supervision without regional fine-tuning
- **Training Data**: Hyogo patches with both reference height TIF and GEDI supervision
- **Model**: Ensemble combining GEDI shift-aware model + reference height model
- **Validation**: Apply ensemble model directly to Tochigi and Kochi regions (no adaptation)
- **Key Question**: Does combined supervision improve generalization without fine-tuning?

### Scenario 3: Reference + GEDI Training + Target Region Adaptation
**Objective**: Evaluate full pipeline with cross-region GEDI fine-tuning
- **Training Data**: Hyogo patches with both reference height TIF and GEDI supervision
- **Model**: Ensemble with GEDI component fine-tuned on target region GEDI data
- **Validation**: Apply adapted ensemble models to Tochigi and Kochi regions
- **Key Question**: Does target region adaptation provide additional performance gains?

## Detailed Implementation for Each Scenario

### Scenario 1 Implementation: Reference-Only Training

#### S1.1 Reference-Only Model Training
**File**: `train_predict_map.py`
- **Model**: 2D U-Net (no temporal dimension, no shift loss, no GEDI)
- **Data**: Hyogo patches (05LE4) with reference height TIF supervision only
- **Loss**: Standard regression loss (MSE/Huber) on all valid pixels
- **Output**: `scenario1_reference_only_model_hyogo.pth`

```bash
python train_predict_map.py \
  --patch-dir chm_outputs/ \
  --include-pattern "*05LE4*" \
  --model 2d_unet \
  --reference-height-path downloads/dchm_05LE4.tif \
  --supervision-mode reference_only \
  --output-dir chm_outputs/scenario1_reference_only \
  --epochs 100 \
  --learning-rate 0.0001 \
  --batch-size 4
```

#### S1.2 Direct Cross-Region Application
**Apply trained model directly to target regions without any adaptation**

```bash
# Apply to Tochigi
python predict_reference_only.py \
  --model-path chm_outputs/scenario1_reference_only/scenario1_reference_only_model_hyogo.pth \
  --patch-dir chm_outputs/ \
  --include-pattern "*09gd4*" \
  --output-dir chm_outputs/scenario1_tochigi_predictions

# Apply to Kochi
python predict_reference_only.py \
  --model-path chm_outputs/scenario1_reference_only/scenario1_reference_only_model_hyogo.pth \
  --patch-dir chm_outputs/ \
  --include-pattern "*04hf3*" \
  --output-dir chm_outputs/scenario1_kochi_predictions
```

### Scenario 2 Implementation: Reference + GEDI Training (No Adaptation)

#### S2.1 Component Model Training
**Train both GEDI shift-aware and reference height models**

**GEDI Shift-Aware Model**:
```bash
python train_predict_map.py \
  --patch-dir chm_outputs/ \
  --include-pattern "*05LE4*" \
  --model shift_aware_unet \
  --shift-radius 2 \
  --output-dir chm_outputs/scenario2_gedi_shift_aware \
  --epochs 100 \
  --learning-rate 0.0001 \
  --batch-size 4
```

**Reference Height Model**:
```bash
python train_predict_map.py \
  --patch-dir chm_outputs/ \
  --include-pattern "*05LE4*" \
  --model 2d_unet \
  --reference-height-path downloads/dchm_05LE4.tif \
  --supervision-mode reference \
  --output-dir chm_outputs/scenario2_reference_2d_unet \
  --epochs 100 \
  --learning-rate 0.0001 \
  --batch-size 4
```

#### S2.2 Ensemble Model Training
**Train MLP ensemble combining both models**

```bash
python train_ensemble_mlp.py \
  --gedi-model-path chm_outputs/scenario2_gedi_shift_aware/gedi_shift_aware_model_hyogo.pth \
  --reference-model-path chm_outputs/scenario2_reference_2d_unet/reference_2d_unet_model_hyogo.pth \
  --patch-dir chm_outputs/ \
  --include-pattern "*05LE4*" \
  --reference-height-path downloads/dchm_05LE4.tif \
  --output-dir chm_outputs/scenario2_ensemble_mlp \
  --epochs 100 \
  --learning-rate 0.001
```

#### S2.3 Direct Cross-Region Application (No Adaptation)
**Apply ensemble model directly to target regions without fine-tuning**

```bash
# Apply to Tochigi
python predict_ensemble.py \
  --gedi-model chm_outputs/scenario2_gedi_shift_aware/gedi_shift_aware_model_hyogo.pth \
  --reference-model chm_outputs/scenario2_reference_2d_unet/reference_2d_unet_model_hyogo.pth \
  --ensemble-mlp chm_outputs/scenario2_ensemble_mlp/ensemble_mlp_model_hyogo.pth \
  --patch-dir chm_outputs/ \
  --include-pattern "*09gd4*" \
  --output-dir chm_outputs/scenario2_tochigi_predictions

# Apply to Kochi
python predict_ensemble.py \
  --gedi-model chm_outputs/scenario2_gedi_shift_aware/gedi_shift_aware_model_hyogo.pth \
  --reference-model chm_outputs/scenario2_reference_2d_unet/reference_2d_unet_model_hyogo.pth \
  --ensemble-mlp chm_outputs/scenario2_ensemble_mlp/ensemble_mlp_model_hyogo.pth \
  --patch-dir chm_outputs/ \
  --include-pattern "*04hf3*" \
  --output-dir chm_outputs/scenario2_kochi_predictions
```

### Scenario 3 Implementation: Reference + GEDI Training + Target Adaptation

#### S3.1 Initial Training (Same as Scenario 2)
**Use same component models and ensemble from Scenario 2**

#### S3.2 Target Region GEDI Adaptation
**Fine-tune GEDI component with target region GEDI data**

**Tochigi GEDI Adaptation**:
```bash
python train_predict_map.py \
  --patch-dir chm_outputs/ \
  --include-pattern "*09gd4*" \
  --model shift_aware_unet \
  --shift-radius 2 \
  --pretrained-model-path chm_outputs/scenario2_gedi_shift_aware/gedi_shift_aware_model_hyogo.pth \
  --output-dir chm_outputs/scenario3_tochigi_gedi_adaptation \
  --epochs 20 \
  --learning-rate 0.00005 \
  --batch-size 4 \
  --adaptation-mode
```

**Kochi GEDI Adaptation**:
```bash
python train_predict_map.py \
  --patch-dir chm_outputs/ \
  --include-pattern "*04hf3*" \
  --model shift_aware_unet \
  --shift-radius 2 \
  --pretrained-model-path chm_outputs/scenario2_gedi_shift_aware/gedi_shift_aware_model_hyogo.pth \
  --output-dir chm_outputs/scenario3_kochi_gedi_adaptation \
  --epochs 20 \
  --learning-rate 0.00005 \
  --batch-size 4 \
  --adaptation-mode
```

#### S3.3 Adapted Ensemble Inference
**Apply ensemble with adapted GEDI models**

```bash
# Tochigi with adapted GEDI model
python predict_ensemble.py \
  --gedi-model chm_outputs/scenario3_tochigi_gedi_adaptation/adapted_gedi_model_tochigi.pth \
  --reference-model chm_outputs/scenario2_reference_2d_unet/reference_2d_unet_model_hyogo.pth \
  --ensemble-mlp chm_outputs/scenario2_ensemble_mlp/ensemble_mlp_model_hyogo.pth \
  --patch-dir chm_outputs/ \
  --include-pattern "*09gd4*" \
  --output-dir chm_outputs/scenario3_tochigi_predictions

# Kochi with adapted GEDI model
python predict_ensemble.py \
  --gedi-model chm_outputs/scenario3_kochi_gedi_adaptation/adapted_gedi_model_kochi.pth \
  --reference-model chm_outputs/scenario2_reference_2d_unet/reference_2d_unet_model_hyogo.pth \
  --ensemble-mlp chm_outputs/scenario2_ensemble_mlp/ensemble_mlp_model_hyogo.pth \
  --patch-dir chm_outputs/ \
  --include-pattern "*04hf3*" \
  --output-dir chm_outputs/scenario3_kochi_predictions
## Comprehensive Validation Framework

### Validation Strategy for All Scenarios
**Objective**: Compare performance across all three scenarios using consistent metrics

#### Validation Data
- **Tochigi**: Reference TIF file `downloads/dchm_09gd4.tif` (for evaluation only)
- **Kochi**: Reference TIF file `downloads/dchm_04hf3.tif` (for evaluation only)
- **Hyogo**: Reference TIF file `downloads/dchm_05LE4.tif` (training supervision + evaluation)

#### Validation Pipeline
```bash
# Scenario 1: Reference-Only Validation
python validate_scenario_comparison.py \
  --scenario 1 \
  --predictions-tochigi chm_outputs/scenario1_tochigi_predictions/ \
  --predictions-kochi chm_outputs/scenario1_kochi_predictions/ \
  --reference-tochigi downloads/dchm_09gd4.tif \
  --reference-kochi downloads/dchm_04hf3.tif \
  --output-dir chm_outputs/validation_scenario1

# Scenario 2: Reference + GEDI (No Adaptation) Validation
python validate_scenario_comparison.py \
  --scenario 2 \
  --predictions-tochigi chm_outputs/scenario2_tochigi_predictions/ \
  --predictions-kochi chm_outputs/scenario2_kochi_predictions/ \
  --reference-tochigi downloads/dchm_09gd4.tif \
  --reference-kochi downloads/dchm_04hf3.tif \
  --output-dir chm_outputs/validation_scenario2

# Scenario 3: Reference + GEDI + Adaptation Validation
python validate_scenario_comparison.py \
  --scenario 3 \
  --predictions-tochigi chm_outputs/scenario3_tochigi_predictions/ \
  --predictions-kochi chm_outputs/scenario3_kochi_predictions/ \
  --reference-tochigi downloads/dchm_09gd4.tif \
  --reference-kochi downloads/dchm_04hf3.tif \
  --output-dir chm_outputs/validation_scenario3

# Comprehensive Comparison Across All Scenarios
python compare_all_scenarios.py \
  --scenario1-results chm_outputs/validation_scenario1/ \
  --scenario2-results chm_outputs/validation_scenario2/ \
  --scenario3-results chm_outputs/validation_scenario3/ \
  --output-dir chm_outputs/comprehensive_scenario_comparison
```

## Technical Implementation Details

### Data Flow Architecture for Three-Scenario Comparison

```
Training Data (Hyogo Region):
├── Satellite Patches (05LE4) - Input features for all scenarios
├── GEDI Points (05LE4) - Sparse supervision for Scenarios 2 & 3
└── Reference TIF (dchm_05LE4.tif) - Dense supervision for all scenarios

┌─────────────────────────────────────────────────────────────────────────────┐
│ Scenario 1: Reference-Only Training                                        │
│ Satellite Patches + Reference TIF -> 2D U-Net -> Reference-Only Model     │
│ Apply Directly to Target Regions (No Adaptation)                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Scenario 2: Reference + GEDI Training (No Adaptation)                      │
│ ├── GEDI Branch: Satellite + GEDI -> Shift-Aware U-Net                     │
│ ├── Reference Branch: Satellite + Reference TIF -> 2D U-Net                │
│ └── Ensemble: Both Outputs -> MLP Ensemble                                 │
│ Apply Directly to Target Regions (No Adaptation)                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Scenario 3: Reference + GEDI Training + Target Adaptation                  │
│ ├── Initial Training: Same as Scenario 2                                   │
│ ├── Target GEDI Adaptation:                                                │
│ │   ├── Tochigi: Fine-tune GEDI Model with GEDI (09gd4)                   │
│ │   └── Kochi: Fine-tune GEDI Model with GEDI (04hf3)                     │
│ └── Ensemble: Adapted GEDI + Frozen Reference + Frozen MLP                 │
└─────────────────────────────────────────────────────────────────────────────┘

Cross-Region Validation:
├── Tochigi: All Scenario Predictions -> Reference TIF (dchm_09gd4.tif) -> Metrics
├── Kochi: All Scenario Predictions -> Reference TIF (dchm_04hf3.tif) -> Metrics
└── Comparison: Scenario 1 vs 2 vs 3 Performance Analysis
```

### File Modifications Required

#### High Priority (Core Implementation)
1. **`data/multi_patch.py`**: Enhanced data loading for reference height supervision
2. **`train_predict_map.py`**: Extended training pipeline with reference height support and adaptation mode
3. **`models/ensemble_mlp.py`**: NEW - Ensemble MLP architecture combining both models
4. **`train_ensemble_mlp.py`**: NEW - Training pipeline for ensemble model
5. **`predict_reference_only.py`**: NEW - Inference pipeline for Scenario 1 (reference-only)
6. **`predict_ensemble.py`**: NEW - Inference pipeline for Scenarios 2 & 3 (ensemble models)
7. **`validate_scenario_comparison.py`**: NEW - Validation framework for individual scenarios
8. **`compare_all_scenarios.py`**: NEW - Comprehensive comparison across all scenarios

#### Medium Priority (Advanced Features)  
1. **`models/losses/`**: Enhanced loss functions for dense supervision
2. **`models/trainers/`**: Enhanced trainers for ensemble training
3. **`utils/spatial_utils.py`**: Spatial alignment utilities for reference height extraction
4. **`data/reference_height_loader.py`**: NEW - Specialized loader for reference TIF data

#### Low Priority (Optimization)
1. **`config/ensemble_config.py`**: NEW - Configuration management for ensemble training
2. **`utils/ensemble_utils.py`**: NEW - Utility functions for ensemble model operations
3. **`docs/ensemble_training_guide.md`**: NEW - Comprehensive user guide

### Expected Outcomes and Research Hypotheses

#### Scenario-Based Performance Hypotheses

**Scenario 1: Reference-Only Training**
- **Hypothesis**: Dense reference height supervision will achieve high performance on Hyogo but may show degraded generalization to other regions
- **Expected Performance**: 
  - Hyogo: R² > 0.8 (excellent with dense supervision)
  - Tochigi: R² 0.4-0.6 (moderate generalization)
  - Kochi: R² 0.4-0.6 (moderate generalization)
- **Key Insight**: Tests the limits of reference-only training for cross-region generalization

**Scenario 2: Reference + GEDI (No Adaptation)**
- **Hypothesis**: Combined supervision will improve generalization while maintaining high performance
- **Expected Performance**:
  - Hyogo: R² > 0.8 (maintained high performance)
  - Tochigi: R² 0.6-0.7 (improved generalization vs Scenario 1)
  - Kochi: R² 0.6-0.7 (improved generalization vs Scenario 1)
- **Key Insight**: Tests whether GEDI supervision enhances cross-region robustness

**Scenario 3: Reference + GEDI + Adaptation**
- **Hypothesis**: Target region adaptation will provide the best overall performance
- **Expected Performance**:
  - Hyogo: R² > 0.8 (maintained high performance)
  - Tochigi: R² > 0.7 (best performance with adaptation)
  - Kochi: R² > 0.7 (best performance with adaptation)
- **Key Insight**: Tests the value of target region fine-tuning

#### Comparative Analysis Framework
1. **Cross-Region Generalization**: How well does each approach generalize beyond training region?
2. **Data Efficiency**: What is the contribution of each supervision source?
3. **Adaptation Benefits**: Does target region GEDI fine-tuning justify the additional complexity?
4. **Practical Implementation**: Which scenario provides the best trade-off between performance and complexity?

## Implementation Timeline

### Week 1-2: Scenario 1 Implementation
- [ ] **Infrastructure**: Implement reference height data loading (`load_multi_patch_reference_data()`)
- [ ] **Infrastructure**: Enhance `train_predict_map.py` with reference height support
- [ ] **Scenario 1**: Train reference-only 2D U-Net model on Hyogo region
- [ ] **Scenario 1**: Implement `predict_reference_only.py` inference pipeline
- [ ] **Scenario 1**: Apply model to Tochigi and Kochi regions
- [ ] **Scenario 1**: Initial validation and performance assessment

### Week 3-4: Scenario 2 Implementation
- [ ] **Scenario 2**: Train GEDI shift-aware model on Hyogo region
- [ ] **Scenario 2**: Train reference height 2D U-Net on Hyogo region
- [ ] **Scenario 2**: Implement ensemble MLP architecture (`models/ensemble_mlp.py`)
- [ ] **Scenario 2**: Train ensemble model combining both outputs (`train_ensemble_mlp.py`)
- [ ] **Scenario 2**: Implement `predict_ensemble.py` inference pipeline
- [ ] **Scenario 2**: Apply ensemble model to Tochigi and Kochi regions (no adaptation)

### Week 5-6: Scenario 3 Implementation
- [ ] **Scenario 3**: Use Scenario 2 models as starting point
- [ ] **Scenario 3**: Implement GEDI adaptation framework in `train_predict_map.py`
- [ ] **Scenario 3**: Fine-tune GEDI models on Tochigi and Kochi GEDI data
- [ ] **Scenario 3**: Apply adapted ensemble models to target regions
- [ ] **Scenario 3**: Enhanced `predict_ensemble.py` for adapted models

### Week 7-8: Comprehensive Analysis & Documentation
- [ ] **Validation**: Implement `validate_scenario_comparison.py` framework
- [ ] **Validation**: Implement `compare_all_scenarios.py` comprehensive comparison
- [ ] **Analysis**: Run validation for all three scenarios across all regions
- [ ] **Analysis**: Statistical analysis and performance ranking
- [ ] **Documentation**: Results analysis and scenario recommendations
- [ ] **Integration**: Best-performing scenario integration with existing pipeline

## Risk Mitigation

### Technical Risks
1. **Ensemble Model Complexity**: MLP ensemble may overfit or underperform
   - **Solution**: Use dropout, cross-validation, and careful hyperparameter tuning
2. **Model Integration Challenges**: Combining outputs from different architectures
   - **Solution**: Standardize output formats and implement robust feature extraction
3. **Memory Usage**: Training multiple models and ensemble requires significant memory
   - **Solution**: Sequential training approach and memory-efficient implementations
4. **Spatial Misalignment**: Reference TIF and satellite patches may not align perfectly
   - **Solution**: Use existing spatial alignment utilities in `utils/spatial_utils.py`

### Training Risks
1. **GEDI vs Reference Height Performance Gap**: Models may perform very differently
   - **Solution**: Careful ensemble weight initialization and balanced training
2. **GEDI Model Adaptation Drift**: Fine-tuning GEDI model may degrade ensemble performance
   - **Solution**: Conservative learning rates and early stopping based on validation metrics
3. **Frozen Component Mismatch**: Frozen reference height model may not generalize to target regions
   - **Solution**: Ensemble MLP trained to be robust to regional variations in reference model outputs
4. **Limited GEDI Data in Target Regions**: Insufficient GEDI samples for effective adaptation
   - **Solution**: Use existing GEDI filtering thresholds and data augmentation techniques

### Validation Risks
1. **Reference Height Quality Variations**: Different regions may have varying reference data quality
   - **Solution**: Region-specific quality assessment and adaptive thresholds
2. **Temporal Mismatch**: Reference heights may not match satellite acquisition dates
   - **Solution**: Acknowledge temporal differences in evaluation and focus on spatial patterns

## Success Metrics

### Technical Success
- [ ] **Step 1**: GEDI shift-aware model successfully trained on Hyogo (Val loss ~13.33)
- [ ] **Step 2**: Reference height 2D U-Net training pipeline operational with dense supervision
- [ ] **Step 3**: Ensemble MLP successfully combines both model outputs
- [ ] **Step 4**: Cross-region adaptation framework functional for Tochigi and Kochi

### Performance Success
- [ ] **Step 1**: GEDI model performance comparable to existing shift-aware results
- [ ] **Step 2**: Reference height model achieves superior performance on Hyogo (R² > 0.6)
- [ ] **Step 3**: Ensemble model outperforms individual models (R² > 0.7)
- [ ] **Step 4**: Cross-region models maintain performance (R² > 0.5) after adaptation

### Scientific Success
- [ ] Comprehensive comparison of GEDI vs reference height supervision strategies
- [ ] Demonstrated benefits of ensemble approach combining sparse and dense supervision
- [ ] Published methodology for multi-stage ensemble training
- [ ] Improved understanding of cross-region generalization in canopy height modeling

### Validation Success
- [ ] All three regional models validated against their respective reference TIF files
- [ ] Height-stratified analysis shows improved performance across all height ranges
- [ ] Spatial consistency metrics demonstrate reduced artifacts
- [ ] Cross-region performance maintains acceptable accuracy thresholds

## Conclusion

This enhanced plan implements a sophisticated four-step ensemble training strategy that leverages both GEDI shift-aware training and dense reference height supervision. The approach combines the strengths of sparse geolocation-compensated GEDI supervision with dense reference height supervision through an MLP ensemble architecture.

### Key Innovations
1. **Dual-Model Training**: Separate optimization of GEDI shift-aware and reference height models
2. **Ensemble Integration**: MLP combination preserves strengths of both approaches  
3. **Cross-Region Adaptation**: Fine-tuning ensemble with regional GEDI data
4. **Comprehensive Validation**: Reference TIF validation across all three regions

### Technical Foundation
The implementation builds upon proven components:
- Existing patch-based architecture (256×256 pixels, 2.56km physical patches)
- Unified training pipeline in `train_predict_map.py`
- Production-ready shift-aware training (88.0% improvement, radius=2)
- Comprehensive evaluation framework with reference TIF validation

### Expected Impact
- **Hyogo Training**: Dense supervision with 100% pixel coverage vs <0.3% GEDI coverage
- **Ensemble Performance**: Combined model leveraging both sparse and dense supervision strengths
- **Cross-Region Generalization**: Adapted models maintaining performance across Japan and potentially outside of Japan.
- **Scientific Contribution**: Novel methodology for combining heterogeneous supervision sources

This approach positions the system for significant performance improvements while maintaining backward compatibility and enabling comprehensive cross-region validation using the available reference height datasets.