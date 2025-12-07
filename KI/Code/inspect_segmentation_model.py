"""
============================================
INSPECT SEGMENTATION MODEL STRUCTURE
Analyze U-Net architecture for fine-tuning
============================================
"""

import tensorflow as tf
import numpy as np
from huggingface_hub import snapshot_download

MODEL_REPO = 'bombshelll/unet-brain-tumor-segmentation'

print("="*70)
print("🔍 ANALYZING SEGMENTATION MODEL STRUCTURE")
print("="*70)

# Download model from Hugging Face
print("\n1️⃣  Downloading model from Hugging Face...")
try:
    model_path = snapshot_download(repo_id=MODEL_REPO)
    print(f"✅ Model downloaded to: {model_path}")
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    exit(1)

# Load model
print("\n2️⃣  Loading SavedModel...")
try:
    model = tf.saved_model.load(model_path)
    print(f"✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Inspect signatures
print("\n3️⃣  Available signatures:")
for sig_name in model.signatures.keys():
    print(f"   - {sig_name}")

# Get serving function
serving_fn = model.signatures['serving_default']

print("\n4️⃣  Input signature:")
input_sig = serving_fn.structured_input_signature
print(f"   {input_sig}")

print("\n5️⃣  Output signature:")
output_sig = serving_fn.structured_outputs
print(f"   {output_sig}")

# Try to access layer structure
print("\n6️⃣  Attempting to access layer structure...")
try:
    # Check if we can access trackable objects
    if hasattr(model, 'trainable_variables'):
        trainable_vars = model.trainable_variables
        print(f"   Total trainable variables: {len(trainable_vars)}")

        print(f"\n   First 10 variable names:")
        for i, var in enumerate(trainable_vars[:10]):
            print(f"      {i+1}. {var.name} - Shape: {var.shape}")

        print(f"\n   Last 10 variable names:")
        for i, var in enumerate(trainable_vars[-10:]):
            print(f"      {len(trainable_vars)-10+i+1}. {var.name} - Shape: {var.shape}")

        # Count total parameters
        total_params = sum(np.prod(var.shape) for var in trainable_vars)
        print(f"\n   Total parameters: {total_params:,}")
    else:
        print("   ⚠️  Cannot access trainable_variables")
except Exception as e:
    print(f"   ⚠️  Error accessing layers: {e}")

# Identify encoder/decoder structure
print("\n7️⃣  Identifying U-Net structure...")
try:
    if hasattr(model, 'trainable_variables'):
        # Group variables by common prefixes (encoder/decoder blocks)
        var_groups = {}
        for var in trainable_vars:
            # Extract prefix (usually indicates layer/block)
            parts = var.name.split('/')
            if len(parts) > 1:
                prefix = '/'.join(parts[:2])  # First 2 levels
                if prefix not in var_groups:
                    var_groups[prefix] = []
                var_groups[prefix].append(var)

        print(f"   Found {len(var_groups)} variable groups:")
        for i, (prefix, vars_list) in enumerate(sorted(var_groups.items())[:15]):
            param_count = sum(np.prod(v.shape) for v in vars_list)
            print(f"      {i+1}. {prefix}: {len(vars_list)} vars, {param_count:,} params")

        if len(var_groups) > 15:
            print(f"      ... ({len(var_groups) - 15} more groups)")
except Exception as e:
    print(f"   ⚠️  Error grouping variables: {e}")

# Test inference
print("\n8️⃣  Testing inference...")
try:
    # Create dummy input
    input_name = list(serving_fn.structured_input_signature[1].keys())[0]
    dummy_input = tf.random.normal((1, 256, 256, 3))

    result = serving_fn(**{input_name: dummy_input})
    output_key = list(result.keys())[0]
    output = result[output_key]

    print(f"   ✅ Inference successful")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.numpy().min():.3f}, {output.numpy().max():.3f}]")
except Exception as e:
    print(f"   ❌ Error during inference: {e}")

print("\n" + "="*70)
print("📊 SUMMARY & RECOMMENDATIONS")
print("="*70)

print("\n🔬 For Fine-Tuning Strategy:")
print("   1. Check if model has encoder-decoder structure")
print("   2. Typically freeze early encoder layers (first 2-3 blocks)")
print("   3. Train decoder layers + last encoder blocks")
print("   4. Lower learning rate than training from scratch (1e-4 to 1e-5)")

print("\n💡 Next Steps:")
print("   1. Review the variable groups above")
print("   2. Identify encoder vs decoder layers")
print("   3. Decide which layers to freeze")
print("   4. Create fine-tuning script with proper layer freezing")

print("\n" + "="*70)
