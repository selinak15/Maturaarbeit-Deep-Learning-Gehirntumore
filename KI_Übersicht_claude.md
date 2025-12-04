# Anmerkung: Für die Maturitätsarbeit habe ich Claude Command genutzt, das ist ein Überblick über alle Prompts und Anwendungen von Claude, die für den Code dieser Maturitätsarbeit genutzt wurden.


# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a medical imaging evaluation and fine-tuning project for brain tumor analysis using deep learning. The project consists of two main tasks:

1. **Classification Task**: Multi-class brain tumor classification (Glioma, Meningioma, Pituitary)
2. **Segmentation Task**: Pixel-level brain tumor segmentation with U-Net

**Target Environment**: Google Colab with GPU (T4 recommended)
**Data Format**: MATLAB v7.3 files (.mat) with HDF5 structure
**Language**: Mixed (German comments, English code)

## Project Structure

```
/Users/selina.keller/claude/
├── code_ma.py                      # Classification evaluation with TTA + optimization
├── fine_tune_brain_tumor.py        # Classification fine-tuning script
├── segmentation_optimized.py       # Segmentation evaluation (threshold + post-proc)
├── segmentation_post_processing.py # Post-processing utilities
├── inspect_model.py                # Classification model inspector
├── inspect_segmentation_model.py   # Segmentation model inspector
├── check.py                        # MAT to PNG converter
├── fine_tuning_strategy.md         # Segmentation fine-tuning guide
└── CLAUDE.md                       # This file
```

## Key Architecture Patterns

### 1. Two-Phase Evaluation Strategy

Both classification and segmentation use a sophisticated evaluation approach:

**Classification** (`code_ma.py`):
- Phase 1: Raw predictions with TTA (8x augmentations)
- Phase 2: Class-weight optimization using Nelder-Mead (lines 422-444)
- Result: 90% → 91% (TTA) → 99.5% (after fine-tuning)

**Segmentation** (`segmentation_optimized.py`):
- Phase 1: Validation set for threshold optimization per class (lines 348-406)
- Phase 2: Test set evaluation with optimized thresholds + post-processing (lines 408-532)
- Result: 70% → 86.57% Dice (threshold opt + post-proc) → 92-95% expected (fine-tuning)

### 2. Medical-Safe Data Augmentation

**CRITICAL**: Medical images require careful augmentation to avoid anatomically impossible transformations.

**Classification TTA** (`code_ma.py`, lines 167-180):
```python
augmented_images = [
    image,                          # Original
    ImageOps.mirror(image),         # Horizontal flip - OK (brain symmetry)
    image.rotate(5),                # Small rotations - OK (±5-10°)
    brightness_enhancer.enhance(1.1)  # Brightness - OK (scanner variation)
]
```

**Segmentation - LEARNED LESSON**:
- ❌ **DO NOT USE**: 90°, 180°, 270° rotations (creates upside-down brains)
- ❌ **DO NOT USE**: Vertical flips (anatomically impossible)
- ✅ **SAFE**: Horizontal flips, brightness (±10%), zoom (±5%)

Rationale: Initial segmentation TTA with rotations DECREASED performance (70% → 67.5% Dice). Brain orientation matters for medical models.

### 3. Label Standardization Pipeline

Models output inconsistent label formats that must be standardized:

**Problem** (`code_ma.py`, lines 75-88):
- Model outputs: `'glioma'`, `'glioma_tumor'`, `'meningioma_tumor'`
- Filenames: `'gl'`, `'me'`, `'pi'`
- External dataset: Integer labels `1`, `2`, `3`

**Solution**: Comprehensive mapping:
```python
LABEL_MAPPING = {
    'glioma': 'glioma',
    'glioma_tumor': 'glioma',
    # ... handles all variations
}

# Filename parser (lines 94-124)
# Handles: "Te-gl_001.jpg", "glioma_TC201_100.png", etc.
```

### 4. HDF5 MATLAB File Loading

All data is in MATLAB v7.3 format requiring HDF5 reading:

**Standard Pattern** (`segmentation_optimized.py`, lines 31-49):
```python
def load_mat_file(filepath):
    with h5py.File(filepath, 'r') as f:
        if 'cjdata' in f:
            image = np.array(f['cjdata']['image']).T      # Note: .T for transpose
            tumor_mask = np.array(f['cjdata']['tumorMask']).T
            label = np.array(f['cjdata']['label'])[0, 0]  # Extract scalar
            return {'image': image, 'tumorMask': tumor_mask, 'label': int(label)}
```

**Key Details**:
- Always transpose arrays (`.T`) due to MATLAB column-major order
- Labels stored as `[0, 0]` arrays, extract with `[0, 0]`
- Try-except for fallback to older MATLAB formats

### 5. Grayscale-to-RGB Model Wrapper

Pre-trained models expect RGB but data is grayscale:

**Wrapper Pattern** (`segmentation_optimized.py`, lines 110-133):
```python
class GrayscaleToRGBModelWrapper:
    def __init__(self, model):
        self.serving_fn = model.signatures.get('serving_default')
        # Auto-detect input tensor name (avoid hardcoding)
        input_signature = self.serving_fn.structured_input_signature[1]
        self.input_name = list(input_signature.keys())[0]

    def predict(self, x):
        if x.shape[-1] == 1:
            x_rgb = np.repeat(x, 3, axis=-1)  # Convert grayscale to RGB
        # ... rest of prediction
```

**Why**: Avoids fragile hardcoded tensor names like `'keras_tensor_68'`

## Common Commands

### Classification Evaluation

```python
# Run in Google Colab
python code_ma.py
```

**Key Parameters** (lines 54-56):
- `USE_FINETUNED = True` - Switch between original and fine-tuned model
- Data path: `/content/drive/My Drive/Testing`
- Output: `/content/drive/My Drive/Brain_Tumor_Results/`

**Output Files**:
- `brain_tumor_results_complete.csv` - All predictions with probabilities
- `critical_errors.csv` - High-confidence errors (potential label mistakes)
- `evaluation_summary.json` - Complete metrics summary

### Segmentation Evaluation

```python
# Run in Google Colab
python segmentation_optimized.py
```

**Key Parameters** (lines 560-561):
- `USE_POST_PROCESSING = True` - Apply morphological operations
- `USE_OPTIMAL_THRESHOLDS = True` - Use class-specific thresholds
- Data path: `/content/drive/My Drive/Testing_Segmentation/**/*.mat`

**Process**:
1. Balanced sampling (700 per class)
2. 20% validation for threshold tuning
3. 80% test for final evaluation
4. Auto-download JSON results

### Fine-Tuning Classification

```python
# Run in Google Colab with GPU
python fine_tune_brain_tumor.py
```

**Configuration** (lines 9-19):
- Model: `Devarshi/Brain_Tumor_Classification` (Swin Transformer, 4 layers)
- Freeze first 2 layers (lines 102-107)
- Learning rate: `2e-5`
- Epochs: 5 with early stopping
- Training data: 1600 images per class

**Expected Results**: 91% → 99.5% accuracy

### Model Inspection

**Classification**:
```python
python inspect_model.py
```
Outputs layer structure to determine freezing strategy.

**Segmentation**:
```python
# Run in Google Colab
python inspect_segmentation_model.py
```
Analyzes U-Net encoder/decoder structure for fine-tuning.

### Data Conversion

Convert external MAT dataset to PNG for classification:

```python
python check.py
```

Creates filenames: `{tumor_type}_{patient_id}_{file_id}.png`

## Critical Implementation Details

### Balanced Sampling

**Rationale**: Original dataset is imbalanced (varies 30-40% across classes)

**Implementation** (`segmentation_optimized.py`, lines 55-98):
```python
def create_balanced_sample(mat_files, samples_per_class=700):
    # Scan all files for labels
    # Random sample N per class
    # Shuffle combined sample
```

**Result**: Equal representation prevents bias in evaluation metrics.

### Post-Processing Pipeline

**Segmentation Only** (`segmentation_post_processing.py`):

Order matters:
1. Remove small objects (noise removal, min_size=50)
2. Morphological closing (fill gaps, kernel=3)
3. Fill holes (binary_fill_holes)
4. Keep largest component (optional, aggressive mode)
5. Morphological opening (smooth edges, kernel=3)

**Impact**: +16% Dice improvement (70% → 86.57%)

### Threshold Optimization

**Per-Class Strategy** (`segmentation_optimized.py`, lines 273-322):

Different tumor types have different optimal thresholds:
- Meningioma: 0.45 (smooth boundaries, lower threshold captures edges)
- Glioma: 0.52 (irregular borders, higher threshold reduces false positives)
- Pituitary: 0.48 (small central tumors)

**Method**: Grid search on validation set (0.3 to 0.7, 41 steps)

### Memory Management

Both scripts include periodic cleanup:

```python
if successful % 25 == 0:
    tf.keras.backend.clear_session()
    import gc
    gc.collect()
```

**Why**: Google Colab has limited RAM; prevents OOM errors on large datasets.

## Data Paths and Conventions

### Classification Data

**Training/Test Split**:
- Path: `/content/drive/My Drive/Testing`
- Format: `(Te|Tr)-(gl|me|pi)_\d+\.(jpg|png)`
- Examples: `Te-gl_001.jpg`, `Tr-me_045.png`

**External Validation** (converted from MAT):
- Format: `{tumor_type}_{patient_id}_{file_id}.png`
- Examples: `glioma_TC201_100.png`, `meningioma_TC001_005.png`

### Segmentation Data

**Path**: `/content/drive/My Drive/Testing_Segmentation/**/*.mat` (recursive)

**MAT Structure**:
```
cjdata/
  ├── image       # Grayscale brain MRI (H x W)
  ├── tumorMask   # Binary mask (H x W)
  └── label       # Integer: 1=Meningioma, 2=Glioma, 3=Pituitary
```

### Label Mapping

**Consistent Across All Scripts**:
```
1 → Meningioma    (smooth, well-defined boundaries)
2 → Glioma        (irregular, infiltrative)
3 → Pituitary     (small, central location)
```

German names in outputs:
- Meningiom
- Gliom
- Hypophysentumor

## Models Used

### Classification

**Base Model**: `Devarshi/Brain_Tumor_Classification`
- Architecture: Swin Transformer (4 encoder layers)
- Input: 224×224 RGB
- Output: 3 classes (after standardization)

**Fine-Tuned Model**: `/content/drive/My Drive/Brain_Tumor_FineTuned/final_model`
- Frozen: First 2 layers
- Trained: Last 2 layers + classifier head
- Performance: 99.5% accuracy

### Segmentation

**Current Model**: `bombshelll/unet-brain-tumor-segmentation` (via `LiamF1111` fork)
- Architecture: U-Net encoder-decoder
- Input: 256×256 RGB (grayscale converted)
- Output: 256×256 binary mask

**Fine-Tuning Strategy** (see `fine_tuning_strategy.md`):
- Freeze: First 2 encoder blocks
- Train: Last 2 encoder blocks + bottleneck + all decoder
- Expected: 92-95% Dice

## Performance Benchmarks

### Classification

| Method | Accuracy | F1-Score (Macro) |
|--------|----------|------------------|
| Base Model | 90% | 0.89 |
| + TTA (8x) | 91% | 0.90 |
| + Class Weights | 91.5% | 0.91 |
| + Fine-Tuning | 99.5% | 0.995 |
| External Validation | 95% | 0.94 |

### Segmentation

| Method | Dice Score | IoU |
|--------|------------|-----|
| Base Model (threshold=0.5) | 70% | 0.54 |
| + Threshold Optimization | 75% | 0.60 |
| + Post-Processing | 86.57% | 0.76 |
| + Fine-Tuning (expected) | 92-95% | 0.85-0.90 |

## Known Issues and Lessons Learned

### 1. TTA Degraded Segmentation Performance

**Issue**: Initial TTA with rotations reduced Dice from 70% → 67.5%

**Root Cause**: Medical models learn anatomical orientation; rotations create invalid brain positions

**Solution**: Use medical-safe augmentations only (horizontal flip, brightness, zoom)

**Files Affected**: `segmentation_better.py` (deprecated), `segmentation_improvements.py` (deprecated)

### 2. Label Confusion Between Similar Tumors

**Issue**: Glioma vs. Pituitary confusion (32 errors in 290 samples)

**Root Cause**: Both can appear in similar locations; subtle visual differences

**Solution**: Fine-tuning on domain-specific data improved to 99.5%

### 3. Hardcoded Tensor Names

**Issue**: Original code used `'keras_tensor_68'` which breaks on model updates

**Solution**: Auto-detection via `structured_input_signature` (lines 116-121)

### 4. Class Imbalance in Original Dataset

**Issue**: Unequal class distribution (30% vs. 40%) biases metrics

**Solution**: Balanced sampling (700 per class) for fair evaluation

## Extension Points

### Adding New Tumor Types

1. Update `LABEL_MAPPING` in all scripts
2. Update `STANDARD_CLASSES` list
3. Add to `tumor_types` dictionary
4. Retrain or fine-tune models

### Changing Dataset Paths

**Pattern**: All paths defined at top of scripts
- Classification: Line ~131
- Segmentation: Line ~9

### Custom Post-Processing

Extend `SegmentationPostProcessing` class with new methods:
- Keep smallest N components
- Watershed segmentation
- Contour smoothing

### Adding CRF Refinement

Next optimization step for segmentation (87-90% Dice target):
- Use `pydensecrf` library
- Refine boundaries using image intensity
- See `fine_tuning_strategy.md` for details
