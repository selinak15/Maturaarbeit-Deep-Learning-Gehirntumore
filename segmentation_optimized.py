# ============================================
# OPTIMIZED BRAIN TUMOR SEGMENTATION EVALUATION
# Mit Threshold Optimization + Post-Processing
# ============================================

import glob

# Durchsucht auch alle Unterordner nach .mat-Dateien
mat_files = glob.glob('/content/drive/My Drive/Testing_Segmentation/**/*.mat', recursive=True)

print(f"✅ Gefunden: {len(mat_files)} .mat Dateien insgesamt")

import tensorflow as tf
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import cv2

print("🔍 Suche .mat Dateien...\n")

# ============================================
# 1. DATEN LADEN
# ============================================

def load_mat_file(filepath):
    """Lädt MATLAB v7.3 Dateien mit h5py"""
    try:
        with h5py.File(filepath, 'r') as f:
            if 'cjdata' in f:
                cjdata = f['cjdata']
                image = np.array(cjdata['image']).T
                tumor_mask = np.array(cjdata['tumorMask']).T
                label = np.array(cjdata['label'])[0, 0]

                return {
                    'image': image,
                    'tumorMask': tumor_mask,
                    'label': int(label),
                }
            else:
                return None
    except Exception as e:
        return None

# ============================================
# 2. BALANCED SAMPLING
# ============================================

def create_balanced_sample(mat_files, samples_per_class=700):
    """Erstellt eine ausgewogene Stichprobe mit gleicher Anzahl pro Klasse"""
    files_by_class = defaultdict(list)
    tumor_types = {1: "Meningiom", 2: "Gliom", 3: "Hypophysentumor"}

    print("🔍 Analysiere Datensatz für Balanced Sampling...")

    for mat_file in tqdm(mat_files, desc="Scanning Dataset"):
        try:
            with h5py.File(mat_file, 'r') as f:
                if 'cjdata' in f:
                    label = int(np.array(f['cjdata']['label'])[0, 0])
                    files_by_class[label].append(mat_file)
        except:
            continue

    print("\n📊 ORIGINAL VERTEILUNG:")
    print("-" * 50)
    for label, files in sorted(files_by_class.items()):
        percentage = len(files) / len(mat_files) * 100 if mat_files else 0
        print(f"{tumor_types[label]:20s}: {len(files):4d} Bilder ({percentage:5.1f}%)")

    min_class_size = min(len(files) for files in files_by_class.values() if files)
    actual_samples_per_class = min(samples_per_class, min_class_size)

    print(f"\n⚖️ BALANCED SAMPLING:")
    print(f"   Samples pro Klasse: {actual_samples_per_class}")
    print("-" * 50)

    balanced_sample = []
    final_distribution = {}

    for label, files in files_by_class.items():
        if files:
            n_samples = min(actual_samples_per_class, len(files))
            selected = random.sample(files, n_samples)
            balanced_sample.extend(selected)
            final_distribution[label] = n_samples
            print(f"{tumor_types[label]:20s}: {n_samples:4d} ausgewählt")

    random.shuffle(balanced_sample)
    print(f"\n✅ FINALE BALANCED STICHPROBE: {len(balanced_sample)} Bilder")

    return balanced_sample, final_distribution

# ============================================
# 3. MODELL LADEN
# ============================================

print("\n📦 Lade Modell von Hugging Face...")
from huggingface_hub import snapshot_download

repo_path = snapshot_download(repo_id="bombshelll/unet-brain-tumor-segmentation")
model = tf.saved_model.load(repo_path)

class GrayscaleToRGBModelWrapper:
    def __init__(self, model):
        self.model = model
        self.serving_fn = model.signatures.get('serving_default')

        try:
            input_signature = self.serving_fn.structured_input_signature[1]
            self.input_name = list(input_signature.keys())[0]
            print(f"   ✅ Auto-detected input name: {self.input_name}")
        except:
            self.input_name = 'keras_tensor_68'
            print(f"   ⚠️ Using fallback input name: {self.input_name}")

    def predict(self, x):
        if x.shape[-1] == 1:
            x_rgb = np.repeat(x, 3, axis=-1)
        else:
            x_rgb = x

        x_tensor = tf.constant(x_rgb, dtype=tf.float32)
        result = self.serving_fn(**{self.input_name: x_tensor})
        output_key = list(result.keys())[0]
        return result[output_key].numpy()

wrapped_model = GrayscaleToRGBModelWrapper(model)
print("✅ Modell bereit")

# ============================================
# 4. METRIKEN
# ============================================

class SegmentationMetrics:
    @staticmethod
    def dice_coefficient(pred, target, smooth=1e-6):
        pred = pred.flatten()
        target = target.flatten()
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    @staticmethod
    def iou_score(pred, target, smooth=1e-6):
        pred = pred.flatten()
        target = target.flatten()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return (intersection + smooth) / (union + smooth)

    @staticmethod
    def precision_recall(pred, target):
        pred_flat = pred.flatten().astype(int)
        target_flat = target.flatten().astype(int)

        tp = np.sum((pred_flat == 1) & (target_flat == 1))
        fp = np.sum((pred_flat == 1) & (target_flat == 0))
        fn = np.sum((pred_flat == 0) & (target_flat == 1))

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        return precision, recall

    @staticmethod
    def hausdorff_distance(pred, target):
        pred_points = np.argwhere(pred)
        target_points = np.argwhere(target)

        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')

        forward = directed_hausdorff(pred_points, target_points)[0]
        backward = directed_hausdorff(target_points, pred_points)[0]
        return max(forward, backward)

# ============================================
# 5. POST-PROCESSING
# ============================================

class PostProcessing:
    """Post-processing to improve segmentation"""

    @staticmethod
    def remove_small_objects(mask, min_size=50):
        """Entfernt kleine isolierte Regionen (Noise)"""
        labeled_mask, num_features = ndimage.label(mask)
        component_sizes = np.bincount(labeled_mask.ravel())
        too_small = component_sizes < min_size
        too_small_mask = too_small[labeled_mask]
        cleaned_mask = mask.copy()
        cleaned_mask[too_small_mask] = 0
        return cleaned_mask

    @staticmethod
    def fill_holes(mask):
        """Füllt Löcher innerhalb von Tumor-Regionen"""
        filled_mask = ndimage.binary_fill_holes(mask).astype(float)
        return filled_mask

    @staticmethod
    def morphological_closing(mask, kernel_size=3):
        """Schließt kleine Lücken in der Maske"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        return closed_mask.astype(float)

    @staticmethod
    def morphological_opening(mask, kernel_size=3):
        """Entfernt kleine Auswüchse an den Rändern"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        return opened_mask.astype(float)

    @staticmethod
    def keep_largest_component(mask):
        """Behält nur die größte zusammenhängende Komponente"""
        if mask.sum() == 0:
            return mask

        labeled_mask, num_features = ndimage.label(mask)
        if num_features == 0:
            return mask

        component_sizes = np.bincount(labeled_mask.ravel())
        component_sizes[0] = 0  # Ignore background

        largest_component = component_sizes.argmax()
        result_mask = (labeled_mask == largest_component).astype(float)
        return result_mask

    @staticmethod
    def apply_pipeline(pred_mask, aggressive=False):
        """
        Komplette Post-Processing Pipeline

        Args:
            pred_mask: Binary prediction mask
            aggressive: If True, applies more aggressive cleaning
        """
        mask = pred_mask.copy()

        # 1. Remove small noise
        min_size = 100 if aggressive else 50
        mask = PostProcessing.remove_small_objects(mask, min_size=min_size)

        # 2. Morphological closing (fill small gaps)
        kernel_size = 5 if aggressive else 3
        mask = PostProcessing.morphological_closing(mask, kernel_size=kernel_size)

        # 3. Fill holes
        mask = PostProcessing.fill_holes(mask)

        # 4. Optional: Keep only largest component
        if aggressive:
            mask = PostProcessing.keep_largest_component(mask)

        # 5. Morphological opening (smooth edges)
        mask = PostProcessing.morphological_opening(mask, kernel_size=3)

        return mask

# ============================================
# 6. THRESHOLD OPTIMIZATION
# ============================================

def find_optimal_threshold_per_class(validation_results):
    """
    Findet optimalen Threshold für jede Tumorklasse

    Args:
        validation_results: List of dicts with 'soft_mask', 'gt_mask', 'label'

    Returns:
        Dict of optimal thresholds per class
    """
    print("\n🔍 Optimiere Thresholds pro Klasse...")

    thresholds_to_test = np.linspace(0.3, 0.7, 41)
    tumor_types = {1: "Meningiom", 2: "Gliom", 3: "Hypophysentumor"}

    optimal_thresholds = {}
    sm = SegmentationMetrics()

    for label in [1, 2, 3]:
        # Filter results for this class
        class_results = [r for r in validation_results if r['label'] == label]

        if not class_results:
            optimal_thresholds[label] = 0.5
            continue

        best_dice = 0
        best_threshold = 0.5

        for threshold in thresholds_to_test:
            dice_scores = []

            for result in class_results:
                soft_mask = result['soft_mask']
                gt_mask = result['gt_mask']

                binary_mask = (soft_mask > threshold).astype(float)
                dice = sm.dice_coefficient(binary_mask, gt_mask)
                dice_scores.append(dice)

            avg_dice = np.mean(dice_scores)

            if avg_dice > best_dice:
                best_dice = avg_dice
                best_threshold = threshold

        optimal_thresholds[label] = best_threshold
        print(f"   {tumor_types[label]:20s}: Threshold = {best_threshold:.3f} (Dice = {best_dice:.4f})")

    return optimal_thresholds

# ============================================
# 7. EVALUATION MIT ALLEN OPTIMIERUNGEN
# ============================================

def evaluate_optimized(wrapped_model, balanced_files, class_distribution,
                      use_post_processing=True,
                      use_optimal_thresholds=True):
    """
    Evaluation mit Post-Processing und Threshold Optimization
    """
    sm = SegmentationMetrics()

    # Phase 1: Validation set für Threshold Optimization (20% der Daten)
    num_validation = int(len(balanced_files) * 0.2)
    validation_files = balanced_files[:num_validation]
    test_files = balanced_files[num_validation:]

    print(f"\n📊 DATEN-SPLIT:")
    print(f"   Validation: {len(validation_files)} Bilder (für Threshold Optimization)")
    print(f"   Test: {len(test_files)} Bilder (finale Evaluation)")

    validation_results = []
    optimal_thresholds = {1: 0.5, 2: 0.5, 3: 0.5}  # Default

    # Phase 1: Threshold Optimization auf Validation Set
    if use_optimal_thresholds:
        print("\n" + "="*60)
        print("PHASE 1: THRESHOLD OPTIMIZATION")
        print("="*60)

        with tqdm(total=len(validation_files), desc="Validation Set") as pbar:
            for mat_file in validation_files:
                try:
                    data = load_mat_file(mat_file)
                    if data is None:
                        pbar.update(1)
                        continue

                    image = data['image'].astype(np.float32)
                    gt_mask = data['tumorMask'].astype(np.float32)
                    label = data['label']

                    # Normalisiere
                    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

                    # Resize
                    if image.shape != (256, 256):
                        image = tf.image.resize(
                            np.expand_dims(image, axis=-1),
                            (256, 256)
                        ).numpy().squeeze()

                    # Predict
                    input_image = image.reshape(1, 256, 256, 1).astype(np.float32)
                    pred = wrapped_model.predict(input_image)

                    if len(pred.shape) == 4:
                        soft_mask = pred[0, :, :, 0]
                    else:
                        soft_mask = pred.squeeze()

                    # Resize back
                    if soft_mask.shape != gt_mask.shape:
                        soft_mask = tf.image.resize(
                            np.expand_dims(soft_mask, axis=-1),
                            gt_mask.shape,
                            method='bilinear'
                        ).numpy().squeeze()

                    validation_results.append({
                        'soft_mask': soft_mask,
                        'gt_mask': gt_mask,
                        'label': label
                    })

                    pbar.update(1)

                except Exception as e:
                    pbar.update(1)
                    continue

        # Find optimal thresholds
        optimal_thresholds = find_optimal_threshold_per_class(validation_results)

    # Phase 2: Evaluation auf Test Set
    print("\n" + "="*60)
    print("PHASE 2: TEST SET EVALUATION")
    print("="*60)

    tumor_types = {1: "Meningiom", 2: "Gliom", 3: "Hypophysentumor"}

    print(f"\n🚀 STARTE EVALUATION")
    print(f"   Post-Processing: {'✅ AKTIVIERT' if use_post_processing else '❌ DEAKTIVIERT'}")
    print(f"   Optimal Thresholds: {'✅ AKTIVIERT' if use_optimal_thresholds else '❌ DEAKTIVIERT'}")
    if use_optimal_thresholds:
        for label, thresh in optimal_thresholds.items():
            print(f"      {tumor_types[label]}: {thresh:.3f}")
    print()

    all_metrics = {
        'dice': [], 'iou': [],
        'precision': [], 'recall': [],
        'hausdorff': [],
        'per_class': {1: [], 2: [], 3: []}
    }

    successful = 0
    failed = 0
    class_counts = {1: 0, 2: 0, 3: 0}

    with tqdm(total=len(test_files), desc="Test Evaluation") as pbar:
        for i, mat_file in enumerate(test_files):
            try:
                pbar.set_description(f"Processing {i+1}/{len(test_files)}")

                data = load_mat_file(mat_file)
                if data is None:
                    failed += 1
                    pbar.update(1)
                    continue

                image = data['image'].astype(np.float32)
                gt_mask = data['tumorMask'].astype(np.float32)
                label = data['label']
                class_counts[label] += 1

                # Normalisiere
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)

                # Resize
                if image.shape != (256, 256):
                    image_resized = tf.image.resize(
                        np.expand_dims(image, axis=-1),
                        (256, 256)
                    ).numpy().squeeze()
                else:
                    image_resized = image

                # Predict
                input_image = image_resized.reshape(1, 256, 256, 1).astype(np.float32)
                pred = wrapped_model.predict(input_image)

                if len(pred.shape) == 4:
                    soft_mask = pred[0, :, :, 0]
                else:
                    soft_mask = pred.squeeze()

                # Resize back
                if soft_mask.shape != gt_mask.shape:
                    soft_mask = tf.image.resize(
                        np.expand_dims(soft_mask, axis=-1),
                        gt_mask.shape,
                        method='bilinear'
                    ).numpy().squeeze()

                # Apply optimal threshold per class
                if use_optimal_thresholds:
                    threshold = optimal_thresholds[label]
                else:
                    threshold = 0.5

                pred_mask = (soft_mask > threshold).astype(np.float32)

                # Apply post-processing
                if use_post_processing:
                    pred_mask = PostProcessing.apply_pipeline(pred_mask, aggressive=False)

                # Calculate metrics
                dice = sm.dice_coefficient(pred_mask, gt_mask)
                iou = sm.iou_score(pred_mask, gt_mask)
                prec, rec = sm.precision_recall(pred_mask, gt_mask)

                all_metrics['dice'].append(dice)
                all_metrics['iou'].append(iou)
                all_metrics['precision'].append(prec)
                all_metrics['recall'].append(rec)
                all_metrics['per_class'][label].append(dice)

                if pred_mask.sum() > 0 and gt_mask.sum() > 0:
                    h_dist = sm.hausdorff_distance(pred_mask, gt_mask)
                    if h_dist != float('inf'):
                        all_metrics['hausdorff'].append(h_dist)

                successful += 1
                pbar.update(1)

                # Memory cleanup
                if successful % 25 == 0:
                    tf.keras.backend.clear_session()
                    import gc
                    gc.collect()

            except Exception as e:
                print(f"\nFehler bei {Path(mat_file).name}: {str(e)[:100]}")
                failed += 1
                pbar.update(1)
                continue

    print(f"\n✅ Evaluation abgeschlossen!")
    print(f"   Erfolgreich: {successful} Bilder")
    print(f"   Fehlgeschlagen: {failed} Bilder")

    print(f"\n📊 TATSÄCHLICHE EVALUATION-VERTEILUNG:")
    print("-" * 50)
    for label, count in class_counts.items():
        percentage = count / successful * 100 if successful > 0 else 0
        print(f"{tumor_types[label]:20s}: {count:4d} Bilder ({percentage:5.1f}%)")

    return all_metrics, successful, class_counts, optimal_thresholds

# ============================================
# 8. HAUPTPROGRAMM
# ============================================

# Erstelle balanced sample
print("\n" + "="*60)
print("📊 BALANCED SAMPLING")
print("="*60)

balanced_files, class_distribution = create_balanced_sample(mat_files, samples_per_class=700)

if not balanced_files:
    print("❌ Kein balanced sample erstellt")
    balanced_files = mat_files[:300]

# Test load
print("\n🧪 Teste das Laden einer .mat Datei...")
if balanced_files:
    test_file = balanced_files[0]
    data = load_mat_file(test_file)
    if data:
        print(f"✅ Erfolgreich geladen!")
        print(f"   Image Shape: {data['image'].shape}")
        print(f"   Mask Shape: {data['tumorMask'].shape}")

# KONFIGURATION
USE_POST_PROCESSING = True      # Empfohlen: True
USE_OPTIMAL_THRESHOLDS = True   # Empfohlen: True

print("\n" + "="*60)
print("🚀 OPTIMIERTE EVALUATION")
print("="*60)
print(f"Post-Processing: {USE_POST_PROCESSING}")
print(f"Optimal Thresholds: {USE_OPTIMAL_THRESHOLDS}")

metrics, num_successful, actual_class_counts, optimal_thresholds = evaluate_optimized(
    wrapped_model, balanced_files, class_distribution,
    use_post_processing=USE_POST_PROCESSING,
    use_optimal_thresholds=USE_OPTIMAL_THRESHOLDS
)

# ============================================
# 9. ERGEBNISSE
# ============================================

if num_successful > 0:
    print("\n" + "="*70)
    print("📊 EVALUATIONSERGEBNISSE")
    print("="*70)

    tumor_types = {1: "Meningiom", 2: "Gliom", 3: "Hypophysentumor"}

    if metrics['dice']:
        print(f"\n🎯 GESAMT-METRIKEN:")
        print("-" * 50)
        print(f"✅ Dice Coefficient: {np.mean(metrics['dice']):.4f} ± {np.std(metrics['dice']):.4f}")
        print(f"   Min: {np.min(metrics['dice']):.4f}, Max: {np.max(metrics['dice']):.4f}")
        print(f"   Median: {np.median(metrics['dice']):.4f}")

        print(f"\n✅ IoU Score: {np.mean(metrics['iou']):.4f} ± {np.std(metrics['iou']):.4f}")
        print(f"✅ Precision: {np.mean(metrics['precision']):.4f} ± {np.std(metrics['precision']):.4f}")
        print(f"✅ Recall: {np.mean(metrics['recall']):.4f} ± {np.std(metrics['recall']):.4f}")

        if metrics['hausdorff']:
            print(f"\n✅ Hausdorff Distance: {np.mean(metrics['hausdorff']):.2f} ± {np.std(metrics['hausdorff']):.2f} pixels")

        # Per-Class Results
        print(f"\n📊 ERGEBNISSE PRO TUMOR-TYP:")
        print("-" * 60)
        print(f"{'Tumor-Typ':<20} {'n':<6} {'Dice':<12} {'Std':<12}")
        print("-" * 60)

        for label, scores in metrics['per_class'].items():
            if scores:
                mean_dice = np.mean(scores)
                std_dice = np.std(scores)
                n_samples = len(scores)
                print(f"{tumor_types[label]:<20} {n_samples:<6} {mean_dice:<12.4f} {std_dice:<12.4f}")

        # Macro Average
        class_means = [np.mean(scores) for scores in metrics['per_class'].values() if scores]
        if class_means:
            macro_dice = np.mean(class_means)
            print(f"\n🎯 MAKRO-DURCHSCHNITT:")
            print(f"   Makro Dice Score: {macro_dice:.4f}")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Dice Distribution
    axes[0, 0].hist(metrics['dice'], bins=30, edgecolor='black', alpha=0.7, color='blue')
    axes[0, 0].set_title(f'Dice Distribution (Mean: {np.mean(metrics["dice"]):.3f})')
    axes[0, 0].set_xlabel('Dice Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(metrics['dice']), color='red', linestyle='--')
    axes[0, 0].grid(True, alpha=0.3)

    # IoU Distribution
    axes[0, 1].hist(metrics['iou'], bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_title(f'IoU Distribution (Mean: {np.mean(metrics["iou"]):.3f})')
    axes[0, 1].set_xlabel('IoU Score')
    axes[0, 1].axvline(np.mean(metrics['iou']), color='red', linestyle='--')
    axes[0, 1].grid(True, alpha=0.3)

    # Boxplot per Class
    class_dice_data = []
    class_labels = []
    for label in [1, 2, 3]:
        if metrics['per_class'][label]:
            class_dice_data.append(metrics['per_class'][label])
            class_labels.append(tumor_types[label])

    if class_dice_data:
        axes[1, 0].boxplot(class_dice_data, labels=class_labels)
        axes[1, 0].set_title('Dice Scores per Tumor Type')
        axes[1, 0].set_ylabel('Dice Score')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        plt.setp(axes[1, 0].get_xticklabels(), rotation=45)

    # Class Distribution
    class_counts = [actual_class_counts[i] for i in [1, 2, 3]]
    class_names = [tumor_types[i] for i in [1, 2, 3]]
    bars = axes[1, 1].bar(class_names, class_counts, color=['skyblue', 'lightgreen', 'salmon'])
    axes[1, 1].set_title('Sample Distribution')
    axes[1, 1].set_ylabel('Number of Images')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    for bar, count in zip(bars, class_counts):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       str(count), ha='center', va='bottom')

    plt.setp(axes[1, 1].get_xticklabels(), rotation=45)

    plt.suptitle(f'Optimized Segmentation Results (Post-Proc: {USE_POST_PROCESSING}, Opt-Thresh: {USE_OPTIMAL_THRESHOLDS})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Save Results
    import json
    from datetime import datetime

    results_dict = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'optimizations': {
            'post_processing': USE_POST_PROCESSING,
            'optimal_thresholds': USE_OPTIMAL_THRESHOLDS
        },
        'optimal_thresholds_values': {tumor_types[k]: float(v) for k, v in optimal_thresholds.items()},
        'total_images_evaluated': len(metrics['dice']),
        'metrics': {
            'dice': {
                'mean': float(np.mean(metrics['dice'])),
                'std': float(np.std(metrics['dice'])),
                'median': float(np.median(metrics['dice']))
            },
            'iou': {
                'mean': float(np.mean(metrics['iou'])),
                'std': float(np.std(metrics['iou']))
            },
            'per_class': {}
        }
    }

    for label, scores in metrics['per_class'].items():
        if scores:
            results_dict['metrics']['per_class'][tumor_types[label]] = {
                'dice_mean': float(np.mean(scores)),
                'dice_std': float(np.std(scores)),
                'n_samples': len(scores)
            }

    class_means = [np.mean(scores) for scores in metrics['per_class'].values() if scores]
    if class_means:
        results_dict['metrics']['macro_dice'] = float(np.mean(class_means))

    filename = '/content/segmentation_optimized_results.json'
    with open(filename, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n💾 Ergebnisse gespeichert in '{filename}'")

    from google.colab import files
    files.download(filename)

    print("\n" + "="*70)
    print("✅ OPTIMIERTE EVALUATION ABGESCHLOSSEN!")
    print("="*70)
    print(f"📊 Evaluiert: {len(metrics['dice'])} Bilder")
    print(f"🎯 Post-Processing: {'✅' if USE_POST_PROCESSING else '❌'}")
    print(f"🎯 Optimal Thresholds: {'✅' if USE_OPTIMAL_THRESHOLDS else '❌'}")
    print(f"📈 Dice Score: {np.mean(metrics['dice']):.4f}")
    print(f"\n💡 Baseline war ~70% Dice")
    print(f"💡 Erwartete Verbesserung: +5-10% → 75-80% Dice")

else:
    print("\n❌ Keine Bilder konnten evaluiert werden")
