# Google Embedding Scenario 3 Implementation Plan: Target Region Fine-tuning

## Executive Summary

**Scenario 3** implements target region fine-tuning of Google Embedding models to test whether adapting pre-trained GEDI models to the target region (Tochigi) can improve ensemble performance while maintaining identical architecture and hyperparameters to Scenario 2A for fair comparison.

## Current Status & Context

### Previous Google Embedding Results
- **Scenario 1**: ✅ Reference-only MLP (R² = 0.8734, outstanding performance)
- **Scenario 2A**: ✅ Spatial U-Net + MLP ensemble (R² = 0.7844, better cross-region than original)

### Scenario 2A Performance (Baseline for Comparison)
| Region | R² Score | RMSE (m) | Correlation | Training Location |
|--------|----------|----------|-------------|-------------------|
| **Kochi** | -1.82 | 11.27 | 0.352 | Hyogo (cross-region) |
| **Hyogo** | -3.12 | 9.61 | 0.310 | Hyogo (training region) |
| **Tochigi** | -0.91 | 8.93 | 0.536 | Hyogo (cross-region) |

## Scenario 3 Hypothesis

**Core Question**: Given a fixed ensemble MLP (trained on Hyogo reference data), can improving the GEDI component through target region training enhance overall ensemble performance?

**Realistic Scenario**: In practice, we often have reference data only in one region (Hyogo) but want to deploy in other regions (Tochigi). The ensemble MLP is fixed, but we can potentially improve the GEDI model component.

**Two Sub-scenarios**:
- **3A**: Replace GEDI model with one trained from scratch on Tochigi
- **3B**: Replace GEDI model with one fine-tuned from Scenario 2A on Tochigi

**Key Insight**: This simulates the practical case where ensemble retraining is not feasible due to lack of reference data in target regions.

## Required Script Modifications

### Update `train_predict_map.py` for Pre-trained Model Support

**Current Issue**: The script doesn't support loading pre-trained models for fine-tuning.

**Required Changes**:

1. **Add Command Line Arguments** (around line 530):
```python
parser.add_argument('--pretrained-model-path', type=str,
                   help='Path to pre-trained model for fine-tuning')
parser.add_argument('--fine-tune-mode', action='store_true',
                   help='Enable fine-tuning mode (load pre-trained weights)')
```

2. **Update ShiftAwareTrainer Initialization** (around line 121):
```python
trainer = ShiftAwareTrainer(
    shift_radius=args.shift_radius,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    band_selection=getattr(args, 'band_selection', 'all'),
    pretrained_model_path=getattr(args, 'pretrained_model_path', None)
)
```

3. **Modify `models/trainers/shift_aware_trainer.py`**:
```python
def __init__(self, shift_radius=2, learning_rate=0.0001, batch_size=4, 
             band_selection='all', pretrained_model_path=None):
    # ... existing code ...
    self.pretrained_model_path = pretrained_model_path

def train(self, train_patches, val_patches, epochs, output_dir):
    # ... model creation ...
    model = self.create_model(input_channels)
    
    # Load pre-trained weights if specified
    if self.pretrained_model_path and os.path.exists(self.pretrained_model_path):
        print(f"🔄 Loading pre-trained model from: {self.pretrained_model_path}")
        try:
            checkpoint = torch.load(self.pretrained_model_path, map_location=self.device)
            model.load_state_dict(checkpoint)
            print("✅ Pre-trained model loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not load pre-trained model: {e}")
    
    # ... rest of training ...
```

### Alternative Approach (Current Implementation)

The current sbatch script trains from scratch on Tochigi data, which provides domain adaptation but not true fine-tuning. This is still scientifically valid for comparing:
- **Scenario 2A**: Hyogo training → cross-region application
- **Scenario 3**: Tochigi training → target region optimization

## Implementation Strategy: Identical Architecture Approach

### Principle: Fair Comparison
- **Identical Model Architecture**: Same shift-aware U-Net and ensemble MLP
- **Identical Hyperparameters**: Exact same learning rates, batch sizes, epochs
- **Identical Loss Functions**: Same shift-aware loss with radius=2
- **Single Variable Change**: Training region (Hyogo → Tochigi)

### Phase 1: Pre-trained Model Foundation
```bash
# Source models from successful Scenario 2A
Base GEDI U-Net: chm_outputs/google_embedding_scenario2a/gedi_unet_model/shift_aware_unet_r2.pth
Reference MLP: chm_outputs/production_mlp_reference_embedding_best.pth
```

### Phase 2: GEDI Model Variants (Target Region Training)

#### Scenario 3A: GEDI Model Trained from Scratch on Tochigi
```bash
python train_predict_map.py \
  --patch-dir "chm_outputs/" \
  --patch-pattern "*09gd4*embedding*" \
  --model shift_aware_unet \
  --shift-radius 2 \
  --supervision-mode gedi \
  --band-selection embedding \
  --output-dir "chm_outputs/google_embedding_scenario3a/gedi_unet_model" \
  --epochs 50 \
  --learning-rate 0.0001 \
  --batch-size 4 \
  --min-gedi-samples 5 \
  --save-model \
  --verbose
```

#### Scenario 3B: GEDI Model Fine-tuned from Scenario 2A on Tochigi
```bash
python train_predict_map.py \
  --patch-dir "chm_outputs/" \
  --patch-pattern "*09gd4*embedding*" \
  --model shift_aware_unet \
  --shift-radius 2 \
  --supervision-mode gedi \
  --band-selection embedding \
  --pretrained-model-path "chm_outputs/google_embedding_scenario2a/gedi_unet_model/shift_aware_unet_r2.pth" \
  --output-dir "chm_outputs/google_embedding_scenario3b/gedi_unet_model" \
  --epochs 50 \
  --learning-rate 0.0001 \
  --batch-size 4 \
  --min-gedi-samples 5 \
  --save-model \
  --verbose \
  --fine-tune-mode
```

**Critical Parameters** (Matching Scenario 2A exactly):
- **Model**: `shift_aware_unet` (identical architecture)
- **Supervision**: `gedi` (not `gedi_only`)
- **Epochs**: `50` (same as Scenario 2A)
- **Learning Rate**: `0.0001` (same as Scenario 2A)
- **Batch Size**: `4` (same as Scenario 2A)
- **Shift Radius**: `2` (same as Scenario 2A)
- **Min GEDI Samples**: `5` (same as Scenario 2A)

### Phase 3: Use Fixed Ensemble MLP (No Retraining)

**Key Insight**: Use the existing Scenario 2A ensemble MLP (trained on Hyogo) without retraining. This simulates the realistic scenario where reference data is only available in the original training region.

```bash
# Copy the fixed ensemble MLP from Scenario 2A
FIXED_ENSEMBLE_PATH="chm_outputs/google_embedding_scenario2a/ensemble_model/ensemble_mlp_best.pth"

# Copy for both scenarios
cp "${FIXED_ENSEMBLE_PATH}" "chm_outputs/google_embedding_scenario3a/ensemble_model/ensemble_mlp_best.pth"
cp "${FIXED_ENSEMBLE_PATH}" "chm_outputs/google_embedding_scenario3b/ensemble_model/ensemble_mlp_best.pth"
```

**No Ensemble Retraining**: The ensemble weights remain fixed from Scenario 2A. We only test whether better GEDI models can improve overall performance through the existing ensemble combination.

### Phase 4: Cross-Region Prediction (Identical Pipeline)

```bash
python predict_ensemble.py \
  --gedi-model "chm_outputs/google_embedding_scenario3/gedi_unet_model/shift_aware_unet_r2.pth" \
  --reference-model "chm_outputs/production_mlp_reference_embedding_best.pth" \
  --ensemble-mlp "chm_outputs/google_embedding_scenario3/ensemble_model/best_model.pth" \
  --patch-dir "chm_outputs/" \
  --regions "04hf3,05LE4,09gd4" \
  --band-selection embedding \
  --output-dir "chm_outputs/google_embedding_scenario3_predictions/"
```

### Phase 5: Comparative Evaluation

```bash
# Use existing evaluate_google_embedding_scenario1.py for Scenario 2A vs 3 comparison
python evaluate_google_embedding_scenario1.py \
  --google-embedding-dir chm_outputs/google_embedding_scenario3_predictions \
  --original-mlp-dir chm_outputs/google_embedding_scenario2a_predictions \
  --downloads-dir downloads \
  --output-dir chm_outputs/scenario2a_vs_scenario3_comparison \
  --no-bias-correction \
  --max-patches 63
```

**Note**: The script will treat Scenario 2A as "original" and Scenario 3 as "google embedding" for comparison purposes. Both use Google Embedding data, but this allows direct performance comparison.

## Batch Processing Scripts

### Scenario 3A Sbatch File (From Scratch Training)
```bash
#!/bin/bash
#SBATCH --job-name=google_embedding_s3a
#SBATCH --output=logs/%j_google_embedding_s3a.txt
#SBATCH --error=logs/%j_google_embedding_s3a_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Activate Python environment
source chm_env/bin/activate

# --- Configuration ---
PATCH_DIR="chm_outputs/"
PATCH_PATTERN="*09gd4*embedding*"
OUTPUT_ROOT="chm_outputs/google_embedding_scenario3a"

# --- Step 1: Train GEDI Spatial U-Net from Scratch on Tochigi ---
GEDI_UNET_DIR="${OUTPUT_ROOT}/gedi_unet_model"
GEDI_UNET_MODEL_PATH="${GEDI_UNET_DIR}/shift_aware_unet_r2.pth"

echo "--- Starting Step 1: Training GEDI Spatial U-Net from Scratch on Tochigi ---"
python train_predict_map.py \
  --patch-dir "${PATCH_DIR}" \
  --patch-pattern "${PATCH_PATTERN}" \
  --model shift_aware_unet \
  --shift-radius 2 \
  --supervision-mode gedi \
  --band-selection embedding \
  --output-dir "${GEDI_UNET_DIR}" \
  --epochs 50 \
  --learning-rate 0.0001 \
  --batch-size 4 \
  --min-gedi-samples 5 \
  --save-model \
  --verbose

if [ ! -f "${GEDI_UNET_MODEL_PATH}" ]; then
    echo "GEDI U-Net training failed. Exiting." >&2
    exit 1
fi
echo "--- GEDI U-Net training completed successfully. ---"

# --- Step 2: Copy Fixed Ensemble MLP (No Retraining) ---
FIXED_ENSEMBLE_PATH="chm_outputs/google_embedding_scenario2a/ensemble_model/ensemble_mlp_best.pth"
ENSEMBLE_DIR="${OUTPUT_ROOT}/ensemble_model"
mkdir -p "${ENSEMBLE_DIR}"

echo "--- Starting Step 2: Using Fixed Ensemble MLP from Scenario 2A ---"
cp "${FIXED_ENSEMBLE_PATH}" "${ENSEMBLE_DIR}/ensemble_mlp_best.pth"

if [ ! -f "${ENSEMBLE_DIR}/ensemble_mlp_best.pth" ]; then
    echo "Fixed ensemble copy failed. Exiting." >&2
    exit 1
fi
echo "--- Fixed ensemble setup completed successfully. ---"

echo "\n✅ Google Embedding Scenario 3A (from scratch + fixed ensemble) workflow finished."
```

### Scenario 3B Sbatch File (Fine-tuning)

```bash
#!/bin/bash
#SBATCH --job-name=google_embedding_s3b
#SBATCH --output=logs/%j_google_embedding_s3b.txt
#SBATCH --error=logs/%j_google_embedding_s3b_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Activate Python environment
source chm_env/bin/activate

# --- Configuration ---
PATCH_DIR="chm_outputs/"
PATCH_PATTERN="*09gd4*embedding*"
OUTPUT_ROOT="chm_outputs/google_embedding_scenario3b"

# --- Step 1: Fine-tune GEDI Spatial U-Net on Tochigi ---
GEDI_UNET_DIR="${OUTPUT_ROOT}/gedi_unet_model"
GEDI_UNET_MODEL_PATH="${GEDI_UNET_DIR}/shift_aware_unet_r2.pth"
PRETRAINED_GEDI_PATH="chm_outputs/google_embedding_scenario2a/gedi_unet_model/shift_aware_unet_r2.pth"

echo "--- Starting Step 1: Fine-tuning GEDI Spatial U-Net on Tochigi ---"
# Note: Requires script modification for --pretrained-model-path support
python train_predict_map.py \
  --patch-dir "${PATCH_DIR}" \
  --patch-pattern "${PATCH_PATTERN}" \
  --model shift_aware_unet \
  --shift-radius 2 \
  --supervision-mode gedi \
  --band-selection embedding \
  --pretrained-model-path "${PRETRAINED_GEDI_PATH}" \
  --output-dir "${GEDI_UNET_DIR}" \
  --epochs 50 \
  --learning-rate 0.0001 \
  --batch-size 4 \
  --min-gedi-samples 5 \
  --save-model \
  --verbose \
  --fine-tune-mode

if [ ! -f "${GEDI_UNET_MODEL_PATH}" ]; then
    echo "GEDI U-Net fine-tuning failed. Exiting." >&2
    exit 1
fi
echo "--- GEDI U-Net fine-tuning completed successfully. ---"

# --- Step 2: Copy Fixed Ensemble MLP (No Retraining) ---
FIXED_ENSEMBLE_PATH="chm_outputs/google_embedding_scenario2a/ensemble_model/ensemble_mlp_best.pth"
ENSEMBLE_DIR="${OUTPUT_ROOT}/ensemble_model"
mkdir -p "${ENSEMBLE_DIR}"

echo "--- Starting Step 2: Using Fixed Ensemble MLP from Scenario 2A ---"
cp "${FIXED_ENSEMBLE_PATH}" "${ENSEMBLE_DIR}/ensemble_mlp_best.pth"

if [ ! -f "${ENSEMBLE_DIR}/ensemble_mlp_best.pth" ]; then
    echo "Fixed ensemble copy failed. Exiting." >&2
    exit 1
fi
echo "--- Fixed ensemble setup completed successfully. ---"

echo "\n✅ Google Embedding Scenario 3B (fine-tuned + fixed ensemble) workflow finished."
```

### Scenario 3 Evaluation Sbatch File

```bash
#!/bin/bash
#SBATCH --job-name=eval_scenario3
#SBATCH --output=logs/%j_evaluate_scenario3_comparison.txt
#SBATCH --error=logs/%j_evaluate_scenario3_comparison_error.txt
#SBATCH --time=0-1:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=48G
#SBATCH --partition=main

# Create output directory
mkdir -p logs chm_outputs/scenario2a_vs_scenario3_comparison

# Activate environment
source chm_env/bin/activate

echo "🚀 Starting Scenario 3 vs Scenario 2A Evaluation at $(date)"
echo "📊 Comparing:"
echo "   • Scenario 2A (Hyogo training): chm_outputs/google_embedding_scenario2a_predictions/"
echo "   • Scenario 3 (Tochigi fine-tuning): chm_outputs/google_embedding_scenario3_predictions/"

# Run evaluation comparing Scenario 2A vs Scenario 3
python evaluate_google_embedding_scenario1.py \
    --google-embedding-dir chm_outputs/google_embedding_scenario3_predictions \
    --original-mlp-dir chm_outputs/google_embedding_scenario2a_predictions \
    --downloads-dir downloads \
    --output-dir chm_outputs/scenario2a_vs_scenario3_comparison \
    --no-bias-correction \
    --max-patches 63

echo "✅ Scenario 3 evaluation completed at $(date)"

# Show results summary
echo "📊 Results Summary:"
if [ -f "chm_outputs/scenario2a_vs_scenario3_comparison/scenario_comparison.csv" ]; then
    echo "Performance comparison table:"
    head -10 chm_outputs/scenario2a_vs_scenario3_comparison/scenario_comparison.csv
fi

if [ -f "chm_outputs/scenario2a_vs_scenario3_comparison/detailed_evaluation_results.json" ]; then
    echo ""
    echo "📋 Detailed results available in JSON format"
fi

echo "📁 Generated files:"
ls -la chm_outputs/scenario2a_vs_scenario3_comparison/

echo ""
echo "🎯 Key Comparison Points:"
echo "   • Scenario 2A: GEDI U-Net + Reference MLP (trained on Hyogo, applied cross-region)"
echo "   • Scenario 3: GEDI U-Net + Reference MLP (fine-tuned on Tochigi, target region adaptation)"
echo "   • Both use identical Google Embedding (64-band) data and architecture"
echo "   • Evaluation across all three regions: Kochi, Hyogo, Tochigi"
echo "   • Focus on Tochigi performance improvement via target region adaptation"

echo ""
echo "🌍 Expected Results:"
echo "   • Tochigi: Scenario 3 should outperform Scenario 2A (target region advantage)"
echo "   • Hyogo: Scenario 2A may outperform Scenario 3 (original training region)"
echo "   • Kochi: Mixed results expected (cross-region for both scenarios)"

echo "🎉 Scenario 3 Evaluation Pipeline Completed!"
echo "📈 Check results in: chm_outputs/scenario2a_vs_scenario3_comparison/"
echo "⏰ Completed at: $(date)"
```

## Expected Performance Analysis

### Performance Hypothesis
| Scenario | GEDI Model | Ensemble MLP | Tochigi R² | Key Question |
|----------|------------|--------------|------------|--------------|
| **2A** | Hyogo-trained | Hyogo-trained | -0.91 | Baseline cross-region performance |
| **3A** | Tochigi from-scratch | Hyogo-trained (fixed) | **-0.5 to 0.0** | Can target-region GEDI improve ensemble? |
| **3B** | Tochigi fine-tuned | Hyogo-trained (fixed) | **-0.3 to +0.2** | Is fine-tuning better than from-scratch? |

### Success Criteria
- **Minimal Success**: Either 3A or 3B shows Tochigi R² > -0.5 (improvement over Scenario 2A: -0.91)
- **Moderate Success**: 3B (fine-tuned) outperforms 3A (from-scratch) on Tochigi
- **Significant Success**: 3B achieves Tochigi R² > 0.0 (positive performance)
- **Major Success**: Target-region GEDI training provides consistent improvement across scenarios

### Scientific Value
This experiment tests the **fixed ensemble + improved GEDI hypothesis**:
- Can better GEDI models improve ensemble performance when ensemble retraining is not feasible?
- Is fine-tuning superior to training from scratch for target region adaptation?
- What are the practical benefits of target-region GEDI model training in deployment scenarios?

## Implementation Timeline

### Week 1: Setup and Fine-tuning
- **Day 1-2**: Create directory structure and prepare sbatch scripts
- **Day 3-5**: Execute GEDI U-Net fine-tuning on Tochigi data
- **Day 6-7**: Validate fine-tuned model and prepare for ensemble training

### Week 2: Ensemble Training and Prediction
- **Day 1-3**: Train ensemble model on Tochigi data
- **Day 4-5**: Generate cross-region predictions for all three regions
- **Day 6-7**: Initial evaluation and sanity checks

### Week 3: Evaluation and Analysis
- **Day 1-3**: Comprehensive evaluation using existing framework
- **Day 4-5**: Comparative analysis with Scenario 2A results
- **Day 6-7**: Performance analysis and visualization

### Week 4: Documentation and Conclusions
- **Day 1-3**: Document results and create comparison visualizations
- **Day 4-5**: Analysis of target region adaptation effectiveness
- **Day 6-7**: Recommendations for production deployment

## File Organization

### Input Files
```
chm_outputs/google_embedding_scenario2a/
├── gedi_unet_model/shift_aware_unet_r2.pth     # Pre-trained GEDI model
└── ensemble_model/ensemble_mlp_best.pth        # Scenario 2A ensemble (reference)

chm_outputs/production_mlp_reference_embedding_best.pth  # Reference model

downloads/dchm_09gd4.tif  # Tochigi reference height data
```

### Output Structure
```
chm_outputs/google_embedding_scenario3/
├── gedi_unet_model/
│   ├── shift_aware_unet_r2.pth              # Fine-tuned GEDI model
│   └── training_logs/                       # Training progress
├── ensemble_model/
│   ├── ensemble_mlp_best.pth               # Scenario 3 ensemble
│   └── training_logs/                       # Ensemble training logs
└── evaluation/
    └── scenario3_performance_analysis.json  # Performance metrics

chm_outputs/google_embedding_scenario3_predictions/
├── kochi/                                   # Cross-region predictions
├── hyogo/                                   # Cross-region predictions
└── tochigi/                                 # Target region predictions

chm_outputs/scenario2a_vs_scenario3_comparison/
├── performance_comparison.csv               # Quantitative comparison
├── correlation_plots/                       # Visualization comparisons
└── analysis_summary.json                   # Key findings
```

### Batch Scripts
```
sbatch/
├── run_google_embedding_scenario2a_training.sh  # Reference (existing)
├── run_google_embedding_scenario3_training.sh   # NEW - Scenario 3 training
├── predict_google_embedding_scenario3.sh        # NEW - Cross-region prediction
└── evaluate_scenario3_comparison.sh             # NEW - Comparative evaluation (uses existing evaluate_google_embedding_scenario1.py)
```

## Key Advantages of This Approach

1. **Scientific Rigor**: Controlled experiment with single variable change (training region)
2. **Fair Comparison**: Identical architecture, hyperparameters, and evaluation metrics
3. **Existing Infrastructure**: Leverages all existing scripts and evaluation frameworks
4. **Clear Attribution**: Performance differences directly attributable to target region adaptation
5. **Production Relevance**: Tests practical deployment scenario (target region training)

## Risk Mitigation

### Technical Risks
- **Catastrophic Forgetting**: Conservative learning rate (0.0001) and limited epochs (50)
- **Overfitting**: Early stopping and validation monitoring during fine-tuning
- **Training Stability**: Use proven hyperparameters from successful Scenario 2A

### Performance Risks
- **Limited Improvement**: Target region may not provide sufficient additional signal
- **Cross-Region Degradation**: Fine-tuning on Tochigi may hurt Kochi/Hyogo performance
- **Ensemble Failure**: If GEDI model remains poor, ensemble may still ignore GEDI component

### Mitigation Strategies
- **Conservative Fine-tuning**: Short training with early stopping
- **Comprehensive Evaluation**: Monitor all regions, not just target region
- **Fallback Plan**: If Scenario 3 fails, Scenario 2A remains production-ready

## Success Metrics

### Technical Success
- [ ] Successful fine-tuning of GEDI U-Net on Tochigi data
- [ ] Ensemble training convergence without degradation
- [ ] Complete cross-region prediction pipeline execution
- [ ] Comprehensive evaluation framework completion

### Performance Success
- [ ] Tochigi R² improvement over Scenario 2A baseline (-0.91)
- [ ] Maintained or improved cross-region generalization
- [ ] Clear statistical significance in target region performance
- [ ] Ensemble model weights indicate effective GEDI integration

### Scientific Success
- [ ] Clear evidence for/against target region adaptation effectiveness
- [ ] Quantified trade-offs between target optimization and generalization
- [ ] Insights into optimal training strategies for sparse supervision scenarios
- [ ] Production deployment recommendations based on empirical results

## Execution Workflow

### Training and Prediction
1. **Training**: `sbatch sbatch/run_google_embedding_scenario3_training.sh`
2. **Prediction**: `sbatch sbatch/predict_google_embedding_scenario3.sh` 
3. **Evaluation**: `sbatch sbatch/evaluate_scenario3_comparison.sh`

### Evaluation Framework
The evaluation uses the existing `evaluate_google_embedding_scenario1.py` script with parameters:
- **Google Embedding Dir**: Scenario 3 predictions (fine-tuned on Tochigi)
- **Original MLP Dir**: Scenario 2A predictions (trained on Hyogo) 
- **No Bias Correction**: For fair comparison
- **Max Patches**: 63 (all available patches)

This approach provides direct comparison between cross-region application (Scenario 2A) and target region adaptation (Scenario 3) using identical evaluation metrics.

## Conclusion

Scenario 3 provides a rigorous test of target region adaptation while maintaining perfect experimental control through identical architectures and hyperparameters. This approach will definitively answer whether fine-tuning on target region data can overcome the cross-region generalization challenges observed in Scenario 2A, providing clear guidance for production deployment strategies.

The plan leverages the successful Google Embedding Scenario 2A foundation while testing a single, well-motivated hypothesis: that target region adaptation can improve ensemble performance. Results will inform future development priorities and production deployment strategies for the Google Embedding-based canopy height prediction system.