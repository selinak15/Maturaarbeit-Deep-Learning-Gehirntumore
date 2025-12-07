"""
============================================
VISUALIZE SEGMENTATION PREDICTIONS
Compare predicted masks with ground truth
============================================

This script loads a trained segmentation model and visualizes:
1. Original MRI image
2. Ground truth mask
3. Predicted mask
4. Overlay comparison
5. Difference map (errors)
"""

import numpy as np
import tensorflow as tf
import h5py
import glob
import matplotlib.pyplot as plt
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

    # Visualization
    'num_samples': 12,  # Number of samples to visualize
    'samples_per_class': 4,  # Samples per tumor type
    'figsize': (20, 12),
    'save_path': '/content/segmentation_visualization.png',

    # Model input
    'img_size': 256,
}

# ============================================
# 1. DATA LOADING
# ============================================

def load_mat_file(file_path):
    """Load .mat file (handles both v7.3 and older formats)"""
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

def load_sample_images(data_dir, samples_per_class=4):
    """Load sample images from each class"""
    all_files = glob.glob(f"{data_dir}/**/*.mat", recursive=True)

    print(f"📁 Found {len(all_files)} total files")

    # Separate by class
    class_files = {1: [], 2: [], 3: []}
    tumor_types = {1: "Meningiom", 2: "Gliom", 3: "Hypophysentumor"}

    for file in all_files:
        try:
            data = load_mat_file(file)
            if data and data['label'] in [1, 2, 3]:
                class_files[data['label']].append(file)
        except:
            continue

    print(f"\n📊 Files per class:")
    for cls in [1, 2, 3]:
        print(f"   {tumor_types[cls]}: {len(class_files[cls])} files")

    # Sample from each class
    selected_files = []
    for cls in [1, 2, 3]:
        if class_files[cls]:
            n = min(samples_per_class, len(class_files[cls]))
            selected = random.sample(class_files[cls], n)
            selected_files.extend(selected)

    print(f"\n✅ Selected {len(selected_files)} samples for visualization")

    # Load the selected files
    samples = []
    for file in selected_files:
        data = load_mat_file(file)
        if data:
            samples.append(data)

    return samples

# ============================================
# 2. MODEL LOADING
# ============================================

def load_segmentation_model(model_path, base_model_repo):
    """
    Load segmentation model

    Args:
        model_path: Path to fine-tuned model (None to use base model)
        base_model_repo: Hugging Face repo for base model
    """
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
            print(f"   ⚠️  Error loading fine-tuned model: {e}")
            print("   📦 Falling back to base model...")

    # Load base model from Hugging Face
    print(f"\n📥 Loading base model from Hugging Face...")
    from huggingface_hub import snapshot_download

    repo_path = snapshot_download(repo_id=base_model_repo)
    saved_model = tf.saved_model.load(repo_path)

    # Wrap in a simple class for prediction
    class ModelWrapper:
        def __init__(self, saved_model):
            self.infer = saved_model.signatures['serving_default']

        def predict(self, x):
            """x should be shape (batch, 256, 256, 3)"""
            result = self.infer(keras_tensor_68=x)
            return result['output_0'].numpy()

    wrapper = ModelWrapper(saved_model)
    print("   ✅ Base model loaded")

    return wrapper, "base"

# ============================================
# 3. PREDICTION
# ============================================

def predict_mask(model, image, img_size=256):
    """
    Predict segmentation mask for an image

    Args:
        model: Loaded segmentation model
        image: Input image (H, W) grayscale
        img_size: Model input size

    Returns:
        pred_mask: Predicted mask (H, W)
        pred_soft: Soft prediction (H, W) - probabilities
    """
    original_shape = image.shape

    # Normalize
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # Resize to model input size
    if image_norm.shape != (img_size, img_size):
        image_resized = cv2.resize(image_norm, (img_size, img_size),
                                   interpolation=cv2.INTER_LINEAR)
    else:
        image_resized = image_norm

    # Convert to RGB
    image_rgb = np.stack([image_resized, image_resized, image_resized], axis=-1)

    # Add batch dimension
    image_batch = np.expand_dims(image_rgb, axis=0).astype(np.float32)

    # Predict
    pred = model.predict(image_batch)

    # Extract mask
    if len(pred.shape) == 4:
        pred_soft = pred[0, :, :, 0]
    else:
        pred_soft = pred.squeeze()

    # Resize back to original size
    if pred_soft.shape != original_shape:
        pred_soft = cv2.resize(pred_soft, (original_shape[1], original_shape[0]),
                              interpolation=cv2.INTER_LINEAR)

    # Threshold to binary
    pred_mask = (pred_soft > 0.5).astype(np.float32)

    return pred_mask, pred_soft

# ============================================
# 4. VISUALIZATION
# ============================================

def calculate_metrics(pred_mask, gt_mask):
    """Calculate Dice and IoU"""
    intersection = (pred_mask * gt_mask).sum()
    dice = (2. * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-6)

    union = pred_mask.sum() + gt_mask.sum() - intersection
    iou = intersection / (union + 1e-6)

    return dice, iou

def visualize_segmentation(samples, model, config):
    """
    Create comprehensive visualization

    Shows:
    - Row 1: Original MRI images
    - Row 2: Ground truth masks
    - Row 3: Predicted masks
    - Row 4: Overlay (GT=green, Pred=red, Overlap=yellow)
    - Row 5: Difference map (FP=red, FN=blue, TP=white)
    """
    num_samples = len(samples)
    tumor_types = {1: "Meningiom", 2: "Gliom", 3: "Hypophysentumor"}

    print("\n" + "="*70)
    print("GENERATING PREDICTIONS")
    print("="*70)

    # Predict all samples
    predictions = []
    for i, sample in enumerate(samples):
        print(f"   Processing {i+1}/{num_samples}: {sample['filename']}")

        pred_mask, pred_soft = predict_mask(model, sample['image'], config['img_size'])
        dice, iou = calculate_metrics(pred_mask, sample['tumorMask'])

        predictions.append({
            'pred_mask': pred_mask,
            'pred_soft': pred_soft,
            'dice': dice,
            'iou': iou
        })

    print("\n" + "="*70)
    print("CREATING VISUALIZATION")
    print("="*70)

    # Create figure with 5 rows
    fig, axes = plt.subplots(5, num_samples, figsize=config['figsize'])

    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    for i, (sample, pred) in enumerate(zip(samples, predictions)):
        image = sample['image']
        gt_mask = sample['tumorMask']
        pred_mask = pred['pred_mask']
        pred_soft = pred['pred_soft']
        label = sample['label']
        dice = pred['dice']
        iou = pred['iou']

        # Normalize image for display
        img_display = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Row 1: Original Image
        axes[0, i].imshow(img_display, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original\nMRI', fontsize=10, fontweight='bold')
        axes[0, i].set_title(f'{tumor_types[label]}', fontsize=9)

        # Row 2: Ground Truth Mask
        axes[1, i].imshow(img_display, cmap='gray')
        axes[1, i].imshow(gt_mask, cmap='Reds', alpha=0.5 * (gt_mask > 0))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Ground\nTruth', fontsize=10, fontweight='bold')

        # Row 3: Predicted Mask
        axes[2, i].imshow(img_display, cmap='gray')
        axes[2, i].imshow(pred_mask, cmap='Blues', alpha=0.5 * (pred_mask > 0))
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Predicted', fontsize=10, fontweight='bold')

        # Row 4: Overlay Comparison
        # Green = GT only, Red = Pred only, Yellow = Both
        overlay = np.zeros((*img_display.shape, 3))
        overlay[:, :, 0] = img_display  # R channel (background)
        overlay[:, :, 1] = img_display  # G channel
        overlay[:, :, 2] = img_display  # B channel

        # GT in green
        overlay[:, :, 1] = np.where(gt_mask > 0, 1.0, overlay[:, :, 1])
        # Pred in red
        overlay[:, :, 0] = np.where(pred_mask > 0, 1.0, overlay[:, :, 0])
        # Overlap becomes yellow (R+G)

        axes[3, i].imshow(overlay)
        axes[3, i].axis('off')
        if i == 0:
            axes[3, i].set_ylabel('Overlay\n(G=GT, R=Pred)', fontsize=10, fontweight='bold')
        axes[3, i].text(0.5, -0.15, f'Dice: {dice:.3f}',
                       transform=axes[3, i].transAxes,
                       ha='center', fontsize=8, fontweight='bold')

        # Row 5: Difference Map
        # True Positive (TP) = white
        # False Positive (FP) = red
        # False Negative (FN) = blue
        # True Negative (TN) = black

        tp = (pred_mask > 0) & (gt_mask > 0)
        fp = (pred_mask > 0) & (gt_mask == 0)
        fn = (pred_mask == 0) & (gt_mask > 0)

        diff_map = np.zeros((*img_display.shape, 3))
        diff_map[tp] = [1, 1, 1]      # White - correct
        diff_map[fp] = [1, 0, 0]      # Red - false positive
        diff_map[fn] = [0, 0, 1]      # Blue - false negative

        axes[4, i].imshow(diff_map)
        axes[4, i].axis('off')
        if i == 0:
            axes[4, i].set_ylabel('Errors\n(R=FP, B=FN)', fontsize=10, fontweight='bold')
        axes[4, i].text(0.5, -0.15, f'IoU: {iou:.3f}',
                       transform=axes[4, i].transAxes,
                       ha='center', fontsize=8, fontweight='bold')

    # Add legend
    legend_text = (
        "Legend:\n"
        "Row 4: Green=GT only, Red=Pred only, Yellow=Both correct\n"
        "Row 5: White=Correct, Red=False Positive, Blue=False Negative"
    )
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Segmentation Visualization: Predictions vs Ground Truth',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    # Save
    plt.savefig(config['save_path'], dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {config['save_path']}")

    # Calculate summary statistics
    avg_dice = np.mean([p['dice'] for p in predictions])
    avg_iou = np.mean([p['iou'] for p in predictions])

    print(f"\n📊 Summary Statistics:")
    print(f"   Average Dice: {avg_dice:.4f}")
    print(f"   Average IoU:  {avg_iou:.4f}")
    print(f"   Min Dice: {min(p['dice'] for p in predictions):.4f}")
    print(f"   Max Dice: {max(p['dice'] for p in predictions):.4f}")

    plt.show()

    return fig

# ============================================
# 5. DETAILED SINGLE IMAGE VISUALIZATION
# ============================================

def visualize_single_detailed(sample, model, config):
    """
    Detailed visualization for a single image
    Shows probability heatmap and multiple views
    """
    image = sample['image']
    gt_mask = sample['tumorMask']
    label = sample['label']
    tumor_types = {1: "Meningiom", 2: "Gliom", 3: "Hypophysentumor"}

    # Predict
    pred_mask, pred_soft = predict_mask(model, image, config['img_size'])
    dice, iou = calculate_metrics(pred_mask, gt_mask)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    img_display = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # 1. Original Image
    axes[0, 0].imshow(img_display, cmap='gray')
    axes[0, 0].set_title('Original MRI', fontweight='bold')
    axes[0, 0].axis('off')

    # 2. Ground Truth
    axes[0, 1].imshow(img_display, cmap='gray')
    axes[0, 1].imshow(gt_mask, cmap='Reds', alpha=0.6 * (gt_mask > 0))
    axes[0, 1].set_title('Ground Truth Mask', fontweight='bold')
    axes[0, 1].axis('off')

    # 3. Prediction
    axes[0, 2].imshow(img_display, cmap='gray')
    axes[0, 2].imshow(pred_mask, cmap='Blues', alpha=0.6 * (pred_mask > 0))
    axes[0, 2].set_title(f'Predicted Mask\nDice: {dice:.4f}', fontweight='bold')
    axes[0, 2].axis('off')

    # 4. Probability Heatmap
    im = axes[1, 0].imshow(pred_soft, cmap='jet', vmin=0, vmax=1)
    axes[1, 0].set_title('Probability Heatmap', fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 5. Overlay
    overlay = np.zeros((*img_display.shape, 3))
    overlay[:, :, :] = np.stack([img_display] * 3, axis=-1)
    overlay[:, :, 1] = np.where(gt_mask > 0, 1.0, overlay[:, :, 1])
    overlay[:, :, 0] = np.where(pred_mask > 0, 1.0, overlay[:, :, 0])

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay (Green=GT, Red=Pred)', fontweight='bold')
    axes[1, 1].axis('off')

    # 6. Difference Map
    tp = (pred_mask > 0) & (gt_mask > 0)
    fp = (pred_mask > 0) & (gt_mask == 0)
    fn = (pred_mask == 0) & (gt_mask > 0)

    diff_map = np.zeros((*img_display.shape, 3))
    diff_map[tp] = [1, 1, 1]
    diff_map[fp] = [1, 0, 0]
    diff_map[fn] = [0, 0, 1]

    axes[1, 2].imshow(diff_map)
    axes[1, 2].set_title('Error Map\n(White=TP, Red=FP, Blue=FN)', fontweight='bold')
    axes[1, 2].axis('off')

    plt.suptitle(f'Detailed View: {tumor_types[label]} - {sample["filename"]}\n'
                f'Dice: {dice:.4f} | IoU: {iou:.4f}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig

# ============================================
# 6. MAIN EXECUTION
# ============================================

def main():
    print("="*70)
    print("🎨 SEGMENTATION VISUALIZATION")
    print("="*70)

    # Load samples
    samples = load_sample_images(CONFIG['data_dir'], CONFIG['samples_per_class'])

    if not samples:
        print("❌ No samples loaded!")
        return

    # Load model
    model, model_type = load_segmentation_model(
        CONFIG['model_path'],
        CONFIG['base_model_repo']
    )

    print(f"\n📊 Using {model_type} model")

    # Create visualization
    fig = visualize_segmentation(samples, model, CONFIG)

    # Optional: Create detailed view for first sample
    print("\n" + "="*70)
    print("CREATING DETAILED VIEW (First Sample)")
    print("="*70)

    detailed_fig = visualize_single_detailed(samples[0], model, CONFIG)
    detailed_path = CONFIG['save_path'].replace('.png', '_detailed.png')
    detailed_fig.savefig(detailed_path, dpi=150, bbox_inches='tight')
    print(f"✅ Detailed view saved to: {detailed_path}")

    plt.show()

    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*70)

if __name__ == '__main__':
    main()
