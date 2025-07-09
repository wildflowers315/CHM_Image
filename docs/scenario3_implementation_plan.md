# Scenario 3 Implementation Plan: Target Region GEDI Adaptation

## Executive Summary

**Scenario 3** focuses on fine-tuning pre-trained GEDI models with target region (Tochigi) GEDI data to overcome the sparse supervision limitations observed in Scenarios 2A and 2B. This approach tests whether target region adaptation can provide the missing signal needed for effective GEDI-based forest height prediction.

## Current Status & Context

### Previous Scenario Results
- **Scenario 1**: ✅ Reference-only MLP (R² = 0.5026, production-ready)
- **Scenario 2A**: ❌ Spatial U-Net + MLP ensemble (R² = -8.58 to -7.95)
- **Scenario 2B**: ❌ Dual-MLP ensemble (R² = -5.14 to -9.95)

### Key Insight
Both Scenario 2 approaches failed because **sparse GEDI supervision (≈0.3% coverage)** from Hyogo region alone was insufficient. **Scenario 3's hypothesis** is that target region adaptation with Tochigi GEDI data (30+ patches) may provide the domain-specific signal needed for effective ensemble training.

## Implementation Strategy

### Phase 1: Pre-trained Model Preparation

#### Available Models
```bash
# Pre-trained models from previous scenarios
chm_outputs/
├── shift_aware_unet_r2.pth              # Spatial U-Net (Hyogo, R² ≈ 0.16)
├── scenario2b_gedi_mlp/gedi_mlp_best.pth # GEDI MLP (Hyogo, failed)
└── production_mlp_best.pth              # Reference MLP (Hyogo, R² = 0.5026)
```

#### Model Assessment
- **Shift-Aware U-Net**: Baseline R² ≈ 0.16 (poor but improvable)
- **GEDI MLP**: Failed in Scenario 2B but may benefit from target adaptation
- **Reference MLP**: Excellent performance, serves as ensemble anchor

### Phase 2: Dual-Track Fine-tuning

#### Track A: Spatial U-Net Adaptation
```bash
# Fine-tune spatial U-Net on Tochigi GEDI data
python train_predict_map.py \
  --patch-dir chm_outputs/enhanced_patches/ \
  --include-pattern "*09gd4*" \
  --model shift_aware_unet \
  --shift-radius 2 \
  --pretrained-model-path chm_outputs/shift_aware_unet_r2.pth \
  --output-dir chm_outputs/scenario3_tochigi_unet_adaptation \
  --epochs 30 \
  --learning-rate 0.00005 \
  --batch-size 4 \
  --fine-tune-mode
```

**Strategy**:
- **Conservative learning rate** (0.00005) to prevent catastrophic forgetting
- **Limited epochs** (30) to avoid overfitting on sparse target data
- **Validation on reference TIF** to maintain consistency
- **Expected improvement**: R² from 0.16 to 0.25-0.30

#### Track B: Pixel-Level MLP Adaptation
```bash
# Fine-tune GEDI MLP on Tochigi GEDI pixels
python train_production_mlp.py \
  --patch-dir chm_outputs/enhanced_patches/ \
  --include-pattern "*09gd4*" \
  --supervision-mode gedi_only \
  --pretrained-model-path chm_outputs/scenario2b_gedi_mlp/gedi_mlp_best.pth \
  --output-dir chm_outputs/scenario3_tochigi_mlp_adaptation \
  --epochs 50 \
  --learning-rate 0.0001 \
  --batch-size 32 \
  --fine-tune-mode
```

**Strategy**:
- **Pixel-level approach** better suited for sparse supervision
- **Domain adaptation** to Tochigi forest characteristics
- **Higher learning rate** (0.0001) for MLP vs U-Net
- **Expected improvement**: R² from failed to 0.20-0.35

### Phase 3: Dual-Track Ensemble Training

#### Ensemble A: Adapted U-Net + Reference MLP
```bash
# Train ensemble with adapted U-Net + Reference MLP
python train_ensemble_mlp.py \
  --gedi-model-path chm_outputs/scenario3_tochigi_unet_adaptation/adapted_unet_tochigi.pth \
  --reference-model-path chm_outputs/production_mlp_best.pth \
  --patch-dir chm_outputs/enhanced_patches/ \
  --include-pattern "*09gd4*" \
  --reference-height-path downloads/dchm_09gd4.tif \
  --output-dir chm_outputs/scenario3_tochigi_unet_ensemble \
  --epochs 50 \
  --learning-rate 0.001
```

#### Ensemble B: Adapted MLP + Reference MLP
```bash
# Train ensemble with adapted GEDI MLP + Reference MLP
python train_ensemble_mlp.py \
  --gedi-model-path chm_outputs/scenario3_tochigi_mlp_adaptation/adapted_mlp_tochigi.pth \
  --reference-model-path chm_outputs/production_mlp_best.pth \
  --patch-dir chm_outputs/enhanced_patches/ \
  --include-pattern "*09gd4*" \
  --reference-height-path downloads/dchm_09gd4.tif \
  --output-dir chm_outputs/scenario3_tochigi_mlp_ensemble \
  --epochs 50 \
  --learning-rate 0.001
```

### Phase 4: Comprehensive Evaluation

#### Evaluation Framework
```bash
# Evaluate both ensemble approaches
python evaluate_with_crs_transform.py \
  --pred-dir chm_outputs/scenario3_tochigi_[unet|mlp]_predictions/ \
  --ref-tif downloads/dchm_09gd4.tif \
  --region-name tochigi \
  --output-dir chm_outputs/scenario3_tochigi_[unet|mlp]_evaluation
```

## Expected Performance Analysis

### Performance Hypothesis

| Approach | Expected R² | Rationale |
|----------|-------------|-----------|
| **Scenario 3A (Adapted U-Net)** | -5.0 to -2.0 | Target adaptation reduces domain shift |
| **Scenario 3B (Adapted MLP)** | -2.0 to +0.1 | Pixel-level + adaptation optimal for sparse data |
| **Baseline (Scenario 1)** | +0.012 | Reference-only with bias correction |

### Success Criteria
- **Minimal Success**: R² > -5.0 (improvement over Scenario 2A: -7.95)
- **Moderate Success**: R² > -2.0 (improvement over Scenario 2B: -9.95)
- **Significant Success**: R² > 0.0 (approaching positive performance)
- **Major Success**: R² > 0.1 (meaningful improvement over baseline)

## Implementation Steps

### Step 1: Environment Setup
```bash
# Activate environment and prepare directories
source chm_env/bin/activate
mkdir -p chm_outputs/scenario3_tochigi_{unet,mlp}_adaptation
mkdir -p chm_outputs/scenario3_tochigi_{unet,mlp}_ensemble
mkdir -p chm_outputs/scenario3_tochigi_{unet,mlp}_predictions
mkdir -p chm_outputs/scenario3_tochigi_{unet,mlp}_evaluation
```

### Step 2: Model Fine-tuning
```bash
# Submit fine-tuning jobs
sbatch sbatch/scenario3_finetune_unet_tochigi.sh
sbatch sbatch/scenario3_finetune_mlp_tochigi.sh
```

### Step 3: Ensemble Training
```bash
# Submit ensemble training jobs
sbatch sbatch/scenario3_train_unet_ensemble.sh
sbatch sbatch/scenario3_train_mlp_ensemble.sh
```

### Step 4: Evaluation
```bash
# Submit evaluation jobs
sbatch sbatch/scenario3_evaluate_unet.sh
sbatch sbatch/scenario3_evaluate_mlp.sh
```

## Key Advantages of Scenario 3

1. **Domain Adaptation**: Target region data should reduce domain shift
2. **Pre-trained Foundation**: Builds on existing Hyogo knowledge
3. **Dual-Track Approach**: Tests both spatial and pixel-level adaptation
4. **Conservative Training**: Prevents catastrophic forgetting
5. **Comprehensive Evaluation**: Compares with all previous scenarios

## Risk Mitigation

### Technical Risks
- **Catastrophic Forgetting**: Conservative learning rates and limited epochs
- **Overfitting**: Early stopping and validation-based training
- **Sparse Data**: Focus on robust architectures (MLP preferred)

### Performance Risks
- **Minimal Improvement**: Even with adaptation, sparse supervision may be fundamentally limiting
- **Ensemble Failure**: If GEDI models remain poor, ensemble may ignore them (like Scenario 2A)
- **Domain Mismatch**: Tochigi characteristics may not generalize to other regions

## Success Metrics

### Technical Success
- [ ] Successful fine-tuning of both U-Net and MLP models
- [ ] Ensemble training convergence without catastrophic forgetting
- [ ] Complete evaluation pipeline execution

### Performance Success
- [ ] Adapted models show improved individual performance
- [ ] Ensemble models achieve R² > -5.0 (better than Scenario 2)
- [ ] At least one approach achieves R² > -2.0

### Scientific Success
- [ ] Clear analysis of adaptation effectiveness
- [ ] Comparison with all previous scenarios
- [ ] Insights into sparse supervision limitations and solutions

## Next Steps

1. **Immediate**: Implement fine-tuning scripts with adaptation support
2. **Short-term**: Execute dual-track fine-tuning and ensemble training
3. **Medium-term**: Comprehensive evaluation and scenario comparison
4. **Long-term**: Document lessons learned and recommend optimal approach

This plan provides a systematic approach to test whether target region adaptation can overcome the fundamental sparse supervision limitations observed in previous scenarios.