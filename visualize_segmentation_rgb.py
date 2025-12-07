"""
============================================
RGB SEGMENTATION VISUALIZATION
Interactive visualization with color overlays
============================================

Features:
- Full RGB color visualization
- Select specific samples by index or class
- Beautiful color overlays
- Side-by-side comparison
- Export individual images
"""

import numpy as np
import tensorflow as tf
import h5py
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from pathlib import Path
import random

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    # Paths
    'data_dir': '/content/drive/My Drive/Tumor/2',
    'model_path': None,  # Set to your fine-tuned model path, or None for base model
    'base_model_repo': 'bombshelll/unet-brain-tumor-segmentation',

    # Sample selection
    'sample_index': None,  # Specific index, or None for random
    'tumor_class': None,   # Filter by class: 1, 2, 3, or None for any
    'sample_id': None,     # Or specify by filename pattern

    # Visualization
    'figsize': (20, 6),
    'save_path': '/content/segmentation_rgb_example.png',
    'img_size': 256,

    # Color scheme
    'gt_color': [0, 255, 0],      # Green for ground truth
    'pred_color': [255, 0, 255],  # Magenta for prediction
    'overlap_color': [255, 255, 0],  # Yellow for overlap
    'fp_color': [255, 0, 0],      # Red for false positive
    'fn_color': [0, 100, 255],    # Blue for false negative
    'tp_color': [255, 255, 255],  # White for true positive
}

# ============================================
# 1. DATA LOADING
# ============================================

def load_mat_file(file_path):
    """Load .mat file"""
    try:
        with h5py.File(file_path, 'r') as f:
            if 'cjdata' in f:
                image = np.array(f['cjdata']['image']).T
                mask = np.array(f['cjdata']['tumorMask']).T
                label = int(np.array(f['cjdata']['label']).item())
                return {
                    'image': image.astype(np.float32),
                    'tumorMask': mask.astype(np.float32),
                    'label': label,
                    'filename': Path(file_path).name
                }
    except:
        pass

    try:
        from scipy.io import loadmat
        mat = loadmat(file_path)
        if 'cjdata' in mat:
            cjdata = mat['cjdata'][0, 0]
            return {
                'image': cjdata['image'].astype(np.float32),
                'tumorMask': cjdata['tumorMask'].astype(np.float32),
                'label': int(cjdata['label']),
                'filename': Path(file_path).name
            }
    except:
        pass

    return None

def find_and_load_sample(data_dir, config):
    """
    Find and load a specific sample based on config criteria
    """
    all_files = glob.glob(f"{data_dir}/**/*.mat", recursive=True)
    print(f"📁 Found {len(all_files)} total files")

    tumor_types = {1: "Meningiom", 2: "Gliom", 3: "Hypophysentumor"}

    # Filter by class if specified
    if config['tumor_class'] is not None:
        print(f"🔍 Filtering for class {config['tumor_class']} ({tumor_types[config['tumor_class']]})")
        class_files = []
        for file in all_files:
            data = load_mat_file(file)
            if data and data['label'] == config['tumor_class']:
                class_files.append(file)
        all_files = class_files
        print(f"   Found {len(all_files)} files of this class")

    # Filter by filename pattern if specified
    if config['sample_id'] is not None:
        print(f"🔍 Filtering for filename pattern: {config['sample_id']}")
        all_files = [f for f in all_files if config['sample_id'] in Path(f).name]
        print(f"   Found {len(all_files)} matching files")

    if not all_files:
        print("❌ No files match the criteria!")
        return None

    # Select specific index or random
    if config['sample_index'] is not None:
        idx = min(config['sample_index'], len(all_files) - 1)
        selected_file = all_files[idx]
        print(f"📌 Selected sample {idx}: {Path(selected_file).name}")
    else:
        selected_file = random.choice(all_files)
        print(f"🎲 Randomly selected: {Path(selected_file).name}")

    return load_mat_file(selected_file)

# ============================================
# 2. MODEL LOADING
# ============================================

def load_segmentation_model(model_path, base_model_repo):
    """Load segmentation model"""
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)

    if model_path and Path(model_path).exists():
        print(f"\n📦 Loading fine-tuned model from: {model_path}")
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print("   ✅ Fine-tuned model loaded")
            return model, "fine-tuned"
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
            print("   📦 Falling back to base model...")

    print(f"\n📥 Loading base model from Hugging Face...")
    from huggingface_hub import snapshot_download

    repo_path = snapshot_download(repo_id=base_model_repo)
    saved_model = tf.saved_model.load(repo_path)

    class ModelWrapper:
        def __init__(self, saved_model):
            self.infer = saved_model.signatures['serving_default']

        def predict(self, x):
            result = self.infer(keras_tensor_68=x)
            return result['output_0'].numpy()

    wrapper = ModelWrapper(saved_model)
    print("   ✅ Base model loaded")

    return wrapper, "base"

# ============================================
# 3. PREDICTION
# ============================================

def predict_mask(model, image, img_size=256):
    """Predict segmentation mask"""
    original_shape = image.shape

    # Normalize
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # Resize
    if image_norm.shape != (img_size, img_size):
        image_resized = cv2.resize(image_norm, (img_size, img_size),
                                   interpolation=cv2.INTER_LINEAR)
    else:
        image_resized = image_norm

    # Convert to RGB
    image_rgb = np.stack([image_resized, image_resized, image_resized], axis=-1)

    # Predict
    image_batch = np.expand_dims(image_rgb, axis=0).astype(np.float32)
    pred = model.predict(image_batch)

    # Extract
    if len(pred.shape) == 4:
        pred_soft = pred[0, :, :, 0]
    else:
        pred_soft = pred.squeeze()

    # Resize back
    if pred_soft.shape != original_shape:
        pred_soft = cv2.resize(pred_soft, (original_shape[1], original_shape[0]),
                              interpolation=cv2.INTER_LINEAR)

    pred_mask = (pred_soft > 0.5).astype(np.float32)

    return pred_mask, pred_soft

# ============================================
# 4. RGB VISUALIZATION
# ============================================

def create_rgb_overlay(image, mask, color, alpha=0.5):
    """
    Create RGB overlay with colored mask

    Args:
        image: Grayscale image (H, W)
        mask: Binary mask (H, W)
        color: RGB color [R, G, B] (0-255)
        alpha: Transparency (0-1)

    Returns:
        RGB image with overlay (H, W, 3)
    """
    # Normalize image to 0-255
    img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    img_norm = (img_norm * 255).astype(np.uint8)

    # Convert to RGB
    img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)

    # Create colored mask
    colored_mask = np.zeros_like(img_rgb)
    colored_mask[mask > 0] = color

    # Blend
    overlay = img_rgb.copy()
    overlay[mask > 0] = cv2.addWeighted(
        img_rgb[mask > 0],
        1 - alpha,
        colored_mask[mask > 0],
        alpha,
        0
    )

    return overlay

def create_comparison_overlay(image, gt_mask, pred_mask, config):
    """
    Create comparison overlay with different colors for GT and Pred

    Colors:
    - Green: Ground truth only
    - Magenta: Prediction only
    - Yellow: Both (overlap)
    """
    img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    img_norm = (img_norm * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB).astype(np.float32)

    # Create masks for different regions
    gt_only = (gt_mask > 0) & (pred_mask == 0)
    pred_only = (pred_mask > 0) & (gt_mask == 0)
    overlap = (gt_mask > 0) & (pred_mask > 0)

    # Apply colors
    overlay = img_rgb.copy()

    # GT only (green)
    if gt_only.any():
        overlay[gt_only] = 0.5 * img_rgb[gt_only] + 0.5 * np.array(config['gt_color'])

    # Pred only (magenta)
    if pred_only.any():
        overlay[pred_only] = 0.5 * img_rgb[pred_only] + 0.5 * np.array(config['pred_color'])

    # Overlap (yellow)
    if overlap.any():
        overlay[overlap] = 0.5 * img_rgb[overlap] + 0.5 * np.array(config['overlap_color'])

    return overlay.astype(np.uint8)

def create_error_map(image, gt_mask, pred_mask, config):
    """
    Create error map with RGB colors

    Colors:
    - White: True Positive (correct)
    - Red: False Positive (extra prediction)
    - Blue: False Negative (missed tumor)
    - Black: True Negative (background)
    """
    img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    img_norm = (img_norm * 50).astype(np.uint8)  # Dark background
    img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)

    # Calculate error types
    tp = (pred_mask > 0) & (gt_mask > 0)
    fp = (pred_mask > 0) & (gt_mask == 0)
    fn = (pred_mask == 0) & (gt_mask > 0)

    # Apply colors
    error_map = img_rgb.copy()
    error_map[tp] = config['tp_color']
    error_map[fp] = config['fp_color']
    error_map[fn] = config['fn_color']

    return error_map

def visualize_rgb_example(sample, model, config, model_type):
    """
    Create comprehensive RGB visualization for a single example
    """
    image = sample['image']
    gt_mask = sample['tumorMask']
    label = sample['label']
    filename = sample['filename']
    tumor_types = {1: "Meningiom", 2: "Gliom", 3: "Hypophysentumor"}

    print("\n" + "="*70)
    print("GENERATING PREDICTION")
    print("="*70)

    # Predict
    pred_mask, pred_soft = predict_mask(model, image, config['img_size'])

    # Calculate metrics
    intersection = (pred_mask * gt_mask).sum()
    dice = (2. * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-6)
    union = pred_mask.sum() + gt_mask.sum() - intersection
    iou = intersection / (union + 1e-6)

    tp = ((pred_mask > 0) & (gt_mask > 0)).sum()
    fp = ((pred_mask > 0) & (gt_mask == 0)).sum()
    fn = ((pred_mask == 0) & (gt_mask > 0)).sum()
    tn = ((pred_mask == 0) & (gt_mask == 0)).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    print(f"\n📊 Metrics:")
    print(f"   Dice: {dice:.4f}")
    print(f"   IoU:  {iou:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   TP: {tp:.0f}, FP: {fp:.0f}, FN: {fn:.0f}, TN: {tn:.0f}")

    print("\n" + "="*70)
    print("CREATING RGB VISUALIZATION")
    print("="*70)

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=config['figsize'])

    # Normalize image
    img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    img_norm = (img_norm * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)

    # Row 1
    # 1. Original Image
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original MRI', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # 2. Ground Truth Overlay (Green)
    gt_overlay = create_rgb_overlay(image, gt_mask, config['gt_color'], alpha=0.5)
    axes[0, 1].imshow(gt_overlay)
    axes[0, 1].set_title('Ground Truth\n(Green)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # 3. Prediction Overlay (Magenta)
    pred_overlay = create_rgb_overlay(image, pred_mask, config['pred_color'], alpha=0.5)
    axes[0, 2].imshow(pred_overlay)
    axes[0, 2].set_title(f'Prediction\n(Magenta)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # 4. Probability Heatmap
    heatmap = cv2.applyColorMap((pred_soft * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    axes[0, 3].imshow(heatmap)
    axes[0, 3].set_title('Probability Heatmap', fontsize=12, fontweight='bold')
    axes[0, 3].axis('off')

    # Row 2
    # 5. Comparison Overlay
    comparison = create_comparison_overlay(image, gt_mask, pred_mask, config)
    axes[1, 0].imshow(comparison)
    axes[1, 0].set_title('Comparison\nGreen=GT, Magenta=Pred, Yellow=Both',
                        fontsize=10, fontweight='bold')
    axes[1, 0].axis('off')

    # 6. Error Map
    error_map = create_error_map(image, gt_mask, pred_mask, config)
    axes[1, 1].imshow(error_map)
    axes[1, 1].set_title('Error Map\nWhite=TP, Red=FP, Blue=FN',
                        fontsize=10, fontweight='bold')
    axes[1, 1].axis('off')

    # 7. Side-by-side masks
    mask_comparison = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    mask_comparison[:, :image.shape[1]//2] = (np.stack([gt_mask]*3, axis=-1) * 255).astype(np.uint8)[:, :image.shape[1]//2]
    mask_comparison[:, image.shape[1]//2:] = (np.stack([pred_mask]*3, axis=-1) * 255).astype(np.uint8)[:, image.shape[1]//2:]

    # Add dividing line
    mask_comparison[:, image.shape[1]//2-2:image.shape[1]//2+2] = [255, 255, 0]

    axes[1, 2].imshow(mask_comparison)
    axes[1, 2].set_title('Masks Side-by-Side\nLeft=GT, Right=Pred',
                        fontsize=10, fontweight='bold')
    axes[1, 2].axis('off')

    # 8. Metrics text
    axes[1, 3].axis('off')
    metrics_text = (
        f"Tumor Type: {tumor_types[label]}\n"
        f"Filename: {filename}\n"
        f"Model: {model_type}\n\n"
        f"Performance Metrics:\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Dice Coefficient: {dice:.4f}\n"
        f"IoU Score: {iou:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n\n"
        f"Pixel Statistics:\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"True Positive: {tp:.0f}\n"
        f"False Positive: {fp:.0f}\n"
        f"False Negative: {fn:.0f}\n"
        f"True Negative: {tn:.0f}\n\n"
        f"GT Tumor Pixels: {gt_mask.sum():.0f}\n"
        f"Pred Tumor Pixels: {pred_mask.sum():.0f}"
    )
    axes[1, 3].text(0.1, 0.5, metrics_text, fontsize=10,
                   verticalalignment='center',
                   family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add legend
    legend_elements = [
        mpatches.Patch(color=np.array(config['gt_color'])/255, label='Ground Truth'),
        mpatches.Patch(color=np.array(config['pred_color'])/255, label='Prediction'),
        mpatches.Patch(color=np.array(config['overlap_color'])/255, label='Overlap (Correct)'),
        mpatches.Patch(color=np.array(config['fp_color'])/255, label='False Positive'),
        mpatches.Patch(color=np.array(config['fn_color'])/255, label='False Negative'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
              frameon=True, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(f'RGB Segmentation Visualization: {tumor_types[label]} - Dice: {dice:.4f}',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save
    plt.savefig(config['save_path'], dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {config['save_path']}")

    plt.show()

    return fig

# ============================================
# 5. MAIN EXECUTION
# ============================================

def main():
    print("="*70)
    print("🎨 RGB SEGMENTATION VISUALIZATION")
    print("="*70)

    # Find and load sample
    sample = find_and_load_sample(CONFIG['data_dir'], CONFIG)

    if sample is None:
        print("❌ Could not load sample!")
        return

    print(f"\n✅ Loaded sample:")
    print(f"   Filename: {sample['filename']}")
    print(f"   Label: {sample['label']}")
    print(f"   Image shape: {sample['image'].shape}")
    print(f"   Mask shape: {sample['tumorMask'].shape}")
    print(f"   Tumor pixels: {sample['tumorMask'].sum():.0f}")

    # Load model
    model, model_type = load_segmentation_model(
        CONFIG['model_path'],
        CONFIG['base_model_repo']
    )

    # Create visualization
    fig = visualize_rgb_example(sample, model, CONFIG, model_type)

    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\n💡 To view a different example, change CONFIG:")
    print(f"   - sample_index: Specific index (0, 1, 2, ...)")
    print(f"   - tumor_class: Filter by class (1=Meningiom, 2=Gliom, 3=Pituitary)")
    print(f"   - sample_id: Filter by filename pattern (e.g., 'TC201')")

if __name__ == '__main__':
    main()
