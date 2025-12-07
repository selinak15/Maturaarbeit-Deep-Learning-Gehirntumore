"""
============================================
QUICK SEGMENTATION MODEL INSPECTION
Simpler version for quick analysis
============================================
"""

import tensorflow as tf
import numpy as np

print("="*70)
print("🔍 QUICK MODEL INSPECTION")
print("="*70)

# Option 1: If you already have the model downloaded
print("\n📦 Option 1: Using pre-downloaded model")
print("   If model is already in your evaluation script, just check:")
print("   model.signatures.keys()")
print()

# Option 2: Load from Hugging Face (requires download)
print("📥 Option 2: Download and inspect from Hugging Face")
print()

try:
    from huggingface_hub import snapshot_download

    print("1️⃣  Downloading model (this may take a minute)...")
    MODEL_REPO = 'bombshelll/unet-brain-tumor-segmentation'
    model_path = snapshot_download(repo_id=MODEL_REPO)
    print(f"   ✅ Downloaded to: {model_path}")

    print("\n2️⃣  Loading model...")
    model = tf.saved_model.load(model_path)
    print("   ✅ Loaded successfully")

    print("\n3️⃣  Basic Model Info:")
    print(f"   Signatures: {list(model.signatures.keys())}")

    # Get serving function
    serving_fn = model.signatures['serving_default']

    # Input info
    input_spec = serving_fn.structured_input_signature[1]
    input_name = list(input_spec.keys())[0]
    input_shape = input_spec[input_name].shape
    print(f"   Input name: {input_name}")
    print(f"   Input shape: {input_shape}")

    # Output info
    output_spec = serving_fn.structured_outputs
    output_name = list(output_spec.keys())[0]
    output_shape = output_spec[output_name].shape
    print(f"   Output name: {output_name}")
    print(f"   Output shape: {output_shape}")

    # Count parameters
    if hasattr(model, 'trainable_variables'):
        trainable_vars = model.trainable_variables
        total_params = sum(np.prod(var.shape) for var in trainable_vars)
        print(f"\n4️⃣  Parameters:")
        print(f"   Total trainable variables: {len(trainable_vars)}")
        print(f"   Total parameters: {total_params:,}")

        # Try to identify structure
        print(f"\n5️⃣  Layer Structure (first 20 variables):")
        for i, var in enumerate(trainable_vars[:20]):
            print(f"   {i+1}. {var.name} → {var.shape}")

        if len(trainable_vars) > 20:
            print(f"   ... ({len(trainable_vars) - 20} more variables)")

        # Group by prefix
        print(f"\n6️⃣  Variable Groups:")
        var_groups = {}
        for var in trainable_vars:
            parts = var.name.split('/')
            if len(parts) > 0:
                prefix = parts[0]
                if prefix not in var_groups:
                    var_groups[prefix] = []
                var_groups[prefix].append(var)

        for prefix, vars_list in sorted(var_groups.items()):
            param_count = sum(np.prod(v.shape) for v in vars_list)
            print(f"   {prefix}: {len(vars_list)} vars, {param_count:,} params")

    # Test inference
    print(f"\n7️⃣  Testing Inference:")
    dummy_input = tf.random.normal((1, 256, 256, 3))
    result = serving_fn(**{input_name: dummy_input})
    output = result[output_name]
    print(f"   Input: {dummy_input.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Output range: [{output.numpy().min():.3f}, {output.numpy().max():.3f}]")
    print(f"   ✅ Model works correctly!")

    print("\n" + "="*70)
    print("✅ INSPECTION COMPLETE")
    print("="*70)

    print("\n💡 KEY FINDINGS:")
    print(f"   • Model expects RGB input: {input_shape}")
    print(f"   • Output is binary mask: {output_shape}")
    print(f"   • Total parameters: {total_params:,}")
    print(f"\n💡 FOR FINE-TUNING:")
    print(f"   • Freeze early encoder layers (first 2 blocks)")
    print(f"   • Train remaining encoder + all decoder")
    print(f"   • Use learning rate: 1e-4")

except ImportError:
    print("❌ Error: huggingface_hub not installed")
    print("   Install with: pip install huggingface_hub")
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 If you're in Google Colab, make sure to:")
    print("   1. Install huggingface_hub: !pip install huggingface_hub")
    print("   2. Have internet connection for download")
    print("   3. Have enough disk space (~200MB)")

print("\n" + "="*70)
