# ============================================
# IMPROVED BRAIN TUMOR SEGMENTATION EVALUATION
# Mit Test-Time Augmentation (TTA)
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
from sklearn.metrics import precision_score, recall_score
from collections import defaultdict
import random
import matplotlib.pyplot as plt

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

                # Extrahiere Daten (HDF5 speichert transponiert)
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
            # Lade nur das Label (schneller)
            with h5py.File(mat_file, 'r') as f:
                if 'cjdata' in f:
                    label = int(np.array(f['cjdata']['label'])[0, 0])
                    files_by_class[label].append(mat_file)
        except:
            continue

    # Zeige Original-Verteilung
    print("\n📊 ORIGINAL VERTEILUNG:")
    print("-" * 50)
    for label, files in sorted(files_by_class.items()):
        percentage = len(files) / len(mat_files) * 100 if mat_files else 0
        print(f"{tumor_types[label]:20s}: {len(files):4d} Bilder ({percentage:5.1f}%)")

    # Bestimme tatsächliche Anzahl pro Klasse
    min_class_size = min(len(files) for files in files_by_class.values() if files)
    actual_samples_per_class = min(samples_per_class, min_class_size)

    print(f"\n⚖️ BALANCED SAMPLING:")
    print(f"   Gewünschte Samples pro Klasse: {samples_per_class}")
    print(f"   Tatsächliche Samples pro Klasse: {actual_samples_per_class}")
    print("-" * 50)

    # Erstelle ausgewogene Stichprobe
    balanced_sample = []
    final_distribution = {}

    for label, files in files_by_class.items():
        if files:
            n_samples = min(actual_samples_per_class, len(files))
            selected = random.sample(files, n_samples)
            balanced_sample.extend(selected)
            final_distribution[label] = n_samples
            print(f"{tumor_types[label]:20s}: {n_samples:4d} ausgewählt")
        else:
            final_distribution[label] = 0
            print(f"{tumor_types[label]:20s}: {0:4d} ausgewählt (keine verfügbar)")

    # Mische die finale Liste
    random.shuffle(balanced_sample)

    print(f"\n✅ FINALE BALANCED STICHPROBE: {len(balanced_sample)} Bilder")
    print(f"   Pro Klasse: {actual_samples_per_class} Bilder")

    return balanced_sample, final_distribution

# ============================================
# 3. MODELL LADEN MIT AUTO-DETECT
# ============================================

print("\n📦 Lade Modell von Hugging Face...")
from huggingface_hub import snapshot_download

repo_path = snapshot_download(repo_id="bombshelll/unet-brain-tumor-segmentation")
model = tf.saved_model.load(repo_path)

class GrayscaleToRGBModelWrapper:
    """Improved wrapper with auto-detection"""
    def __init__(self, model):
        self.model = model
        self.serving_fn = model.signatures.get('serving_default')

        # Auto-detect input name (no hardcoding!)
        try:
            input_signature = self.serving_fn.structured_input_signature[1]
            self.input_name = list(input_signature.keys())[0]
            print(f"   ✅ Auto-detected input name: {self.input_name}")
        except:
            # Fallback to hardcoded
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
# 4. METRIKEN-KLASSE
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
# 5. TEST-TIME AUGMENTATION (TTA)
# ============================================

class SegmentationTTA:
    """Test-Time Augmentation für bessere Segmentierung"""

    @staticmethod
    def apply_augmentations(image, medical_only=True):
        """
        Erstellt augmentierte Versionen des Bildes

        Args:
            medical_only: If True, only uses medically-appropriate augmentations

        Returns: List of (augmented_image, reverse_transform_fn)
        """
        augs = []

        # 1. Original (always included)
        augs.append((
            image,
            lambda x: x
        ))

        if medical_only:
            # MEDICAL-APPROPRIATE AUGMENTATIONS ONLY
            # Brain scans should stay upright, only horizontal flip is natural

            # 2. Horizontal Flip (brain left-right symmetry is valid)
            augs.append((
                np.fliplr(image),
                lambda x: np.fliplr(x)
            ))

            # That's it! Just 2 augmentations for medical images

        else:
            # FULL TTA (all rotations - not recommended for medical)

            # 2. Horizontal Flip
            augs.append((
                np.fliplr(image),
                lambda x: np.fliplr(x)
            ))

            # 3. Vertical Flip
            augs.append((
                np.flipud(image),
                lambda x: np.flipud(x)
            ))

            # 4. Rotation 90°
            augs.append((
                np.rot90(image, k=1),
                lambda x: np.rot90(x, k=-1)
            ))

            # 5. Rotation 180°
            augs.append((
                np.rot90(image, k=2),
                lambda x: np.rot90(x, k=-2)
            ))

            # 6. Rotation 270°
            augs.append((
                np.rot90(image, k=3),
                lambda x: np.rot90(x, k=-3)
            ))

        return augs

    @staticmethod
    def predict_with_tta(model, image, threshold=0.5):
        """
        Führt Prediction mit TTA durch

        Args:
            model: Wrapped segmentation model
            image: Input image (H, W) oder (H, W, 1)
            threshold: Threshold für finale Maske

        Returns:
            final_mask: Averaged prediction mask
            confidence: Pixel-wise confidence
            soft_mask: Soft probability map before thresholding
        """
        # Ensure correct shape
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        h, w = image.shape[:2]

        # Get augmentations (medical_only=True by default)
        augmentations = SegmentationTTA.apply_augmentations(image, medical_only=True)

        all_predictions = []

        for aug_img, reverse_fn in augmentations:
            # Prepare for model
            if len(aug_img.shape) == 2:
                aug_img = np.expand_dims(aug_img, axis=-1)

            # Resize to 256x256
            aug_img_resized = tf.image.resize(
                aug_img,
                (256, 256)
            ).numpy()

            # Add batch dimension
            input_batch = aug_img_resized.reshape(1, 256, 256, 1).astype(np.float32)

            # Predict
            pred = model.predict(input_batch)

            # Extract mask
            if len(pred.shape) == 4:
                pred_mask = pred[0, :, :, 0]
            else:
                pred_mask = pred.squeeze()

            # Resize back to original size
            pred_mask_resized = tf.image.resize(
                np.expand_dims(pred_mask, axis=-1),
                (h, w),
                method='bilinear'
            ).numpy().squeeze()

            # Apply reverse transformation
            pred_mask_reversed = reverse_fn(pred_mask_resized)

            all_predictions.append(pred_mask_reversed)

        # Stack and average
        predictions_stack = np.stack(all_predictions, axis=0)

        # Average prediction (soft mask)
        soft_mask = np.mean(predictions_stack, axis=0)

        # Confidence: inverse of std (lower std = higher confidence)
        confidence = 1.0 - np.std(predictions_stack, axis=0)

        # Apply threshold
        final_mask = (soft_mask > threshold).astype(np.float32)

        return final_mask, confidence, soft_mask

# ============================================
# 6. EVALUATION MIT TTA
# ============================================

def evaluate_with_tta(wrapped_model, balanced_files, class_distribution, use_tta=True):
    """
    Evaluation mit optionalem TTA

    Args:
        use_tta: If True, uses 6x TTA for better results (slower)
    """
    sm = SegmentationMetrics()
    all_metrics = {
        'dice': [], 'iou': [],
        'precision': [], 'recall': [],
        'hausdorff': [],
        'confidence': [],  # NEW: Track confidence
        'per_class': {1: [], 2: [], 3: []}
    }

    tumor_types = {1: "Meningiom", 2: "Gliom", 3: "Hypophysentumor"}

    print(f"\n🚀 STARTE EVALUATION {'MIT TTA (2x Medical-Only)' if use_tta else 'OHNE TTA'}")
    print(f"   Total Dateien: {len(balanced_files)}")
    if use_tta:
        print("   ⚠️ TTA aktiviert: Medical-appropriate (nur Horizontal Flip)")
        print("   ⚠️ Keine Rotationen (schlecht für Hirn-Scans)")
    print()

    successful = 0
    failed = 0
    class_counts = {1: 0, 2: 0, 3: 0}

    with tqdm(total=len(balanced_files), desc="Evaluation") as pbar:
        for i, mat_file in enumerate(balanced_files):
            try:
                pbar.set_description(f"Processing {i+1}/{len(balanced_files)}")

                # Lade Daten
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

                # PREDICTION WITH OR WITHOUT TTA
                if use_tta:
                    pred_mask, confidence_map, soft_mask = SegmentationTTA.predict_with_tta(
                        wrapped_model, image, threshold=0.5
                    )
                    avg_confidence = np.mean(confidence_map)
                    all_metrics['confidence'].append(avg_confidence)
                else:
                    # Original single prediction
                    if image.shape != (256, 256):
                        image_resized = tf.image.resize(
                            np.expand_dims(image, axis=-1),
                            (256, 256)
                        ).numpy().squeeze()
                    else:
                        image_resized = image

                    input_image = image_resized.reshape(1, 256, 256, 1).astype(np.float32)
                    pred = wrapped_model.predict(input_image)

                    if len(pred.shape) == 4:
                        pred_mask = (pred[0, :, :, 0] > 0.5).astype(np.float32)
                    else:
                        pred_mask = (pred.squeeze() > 0.5).astype(np.float32)

                    if pred_mask.shape != gt_mask.shape:
                        pred_mask = tf.image.resize(
                            np.expand_dims(pred_mask, axis=-1),
                            gt_mask.shape,
                            method='nearest'
                        ).numpy().squeeze()

                # Berechne Metriken
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

                # Speicher aufräumen alle 25 Bilder
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

    if use_tta and all_metrics['confidence']:
        print(f"\n📊 TTA Confidence: {np.mean(all_metrics['confidence']):.4f} ± {np.std(all_metrics['confidence']):.4f}")
        print(f"   (Höher = stabilere Vorhersagen)")

    # Zeige tatsächliche Klassenverteilung
    print(f"\n📊 TATSÄCHLICHE EVALUATION-VERTEILUNG:")
    print("-" * 50)
    for label, count in class_counts.items():
        percentage = count / successful * 100 if successful > 0 else 0
        print(f"{tumor_types[label]:20s}: {count:4d} Bilder ({percentage:5.1f}%)")

    return all_metrics, successful, class_counts

# ============================================
# 7. HAUPTPROGRAMM
# ============================================

# Erstelle balanced sample
print("\n" + "="*60)
print("📊 BALANCED SAMPLING")
print("="*60)

balanced_files, class_distribution = create_balanced_sample(mat_files, samples_per_class=700)

if not balanced_files:
    print("❌ Kein balanced sample erstellt - verwende alle verfügbaren Dateien")
    balanced_files = mat_files[:300]  # Fallback

# Test: Lade eine Beispieldatei
print("\n🧪 Teste das Laden einer .mat Datei...")
if balanced_files:
    test_file = balanced_files[0]
    print(f"Teste mit: {test_file}")

    data = load_mat_file(test_file)
    if data:
        print(f"✅ Erfolgreich geladen!")
        print(f"   Image Shape: {data['image'].shape}")
        print(f"   Mask Shape: {data['tumorMask'].shape}")
        print(f"   Label: {data['label']}")

# WÄHLE TTA MODUS
USE_TTA = True  # Ändere auf False für schnellere Evaluation ohne TTA

print("\n" + "="*60)
print(f"🚀 EVALUATION {'MIT' if USE_TTA else 'OHNE'} TTA")
print("="*60)

metrics, num_successful, actual_class_counts = evaluate_with_tta(
    wrapped_model, balanced_files, class_distribution,
    use_tta=USE_TTA
)

# ============================================
# 8. ERGEBNISSE ANZEIGEN
# ============================================

if num_successful > 0:
    print("\n" + "="*70)
    print("📊 EVALUATIONSERGEBNISSE")
    print("="*70)

    tumor_types = {1: "Meningiom", 2: "Gliom", 3: "Hypophysentumor"}

    if metrics['dice']:
        # Gesamt-Metriken
        print(f"\n🎯 GESAMT-METRIKEN:")
        print("-" * 50)
        print(f"✅ Dice Coefficient: {np.mean(metrics['dice']):.4f} ± {np.std(metrics['dice']):.4f}")
        print(f"   Min: {np.min(metrics['dice']):.4f}, Max: {np.max(metrics['dice']):.4f}")
        print(f"   Median: {np.median(metrics['dice']):.4f}")

        print(f"\n✅ IoU Score: {np.mean(metrics['iou']):.4f} ± {np.std(metrics['iou']):.4f}")
        print(f"   Min: {np.min(metrics['iou']):.4f}, Max: {np.max(metrics['iou']):.4f}")
        print(f"   Median: {np.median(metrics['iou']):.4f}")

        print(f"\n✅ Precision: {np.mean(metrics['precision']):.4f} ± {np.std(metrics['precision']):.4f}")
        print(f"✅ Recall: {np.mean(metrics['recall']):.4f} ± {np.std(metrics['recall']):.4f}")

        if metrics['hausdorff']:
            print(f"\n✅ Hausdorff Distance: {np.mean(metrics['hausdorff']):.2f} ± {np.std(metrics['hausdorff']):.2f} pixels")
            print(f"   Median: {np.median(metrics['hausdorff']):.2f} pixels")

        if USE_TTA and metrics['confidence']:
            print(f"\n✅ TTA Confidence: {np.mean(metrics['confidence']):.4f} ± {np.std(metrics['confidence']):.4f}")

        # Per-Class Ergebnisse
        print(f"\n📊 ERGEBNISSE PRO TUMOR-TYP:")
        print("-" * 60)
        print(f"{'Tumor-Typ':<20} {'n':<6} {'Dice':<12} {'Std':<12} {'Min':<8} {'Max':<8}")
        print("-" * 60)

        for label, scores in metrics['per_class'].items():
            if scores:
                mean_dice = np.mean(scores)
                std_dice = np.std(scores)
                min_dice = np.min(scores)
                max_dice = np.max(scores)
                n_samples = len(scores)

                print(f"{tumor_types[label]:<20} {n_samples:<6} {mean_dice:<12.4f} {std_dice:<12.4f} {min_dice:<8.4f} {max_dice:<8.4f}")

        # Berechne Makro-Durchschnitt
        class_means = []
        for label, scores in metrics['per_class'].items():
            if scores:
                class_means.append(np.mean(scores))

        if class_means:
            macro_dice = np.mean(class_means)
            print(f"\n🎯 MAKRO-DURCHSCHNITT (ungewichtet über Klassen):")
            print(f"   Makro Dice Score: {macro_dice:.4f}")
            print(f"   Standard Dice Score: {np.mean(metrics['dice']):.4f}")

    # ============================================
    # 9. VISUALISIERUNG
    # ============================================

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Dice Distribution
    axes[0, 0].hist(metrics['dice'], bins=30, edgecolor='black', alpha=0.7, color='blue')
    axes[0, 0].set_title(f'Dice Coefficient Distribution (n={len(metrics["dice"])})')
    axes[0, 0].set_xlabel('Dice Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(metrics['dice']), color='red', linestyle='--',
                      label=f'Mean: {np.mean(metrics["dice"]):.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. IoU Distribution
    axes[0, 1].hist(metrics['iou'], bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_title('IoU Distribution')
    axes[0, 1].set_xlabel('IoU Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(metrics['iou']), color='red', linestyle='--',
                      label=f'Mean: {np.mean(metrics["iou"]):.3f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Precision vs Recall
    axes[0, 2].scatter(metrics['precision'], metrics['recall'], alpha=0.6)
    axes[0, 2].set_title('Precision vs Recall')
    axes[0, 2].set_xlabel('Precision')
    axes[0, 2].set_ylabel('Recall')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xlim([0, 1])
    axes[0, 2].set_ylim([0, 1])

    # 4. Boxplot aller Metriken
    data_for_box = [metrics['dice'], metrics['iou'],
                   metrics['precision'], metrics['recall']]
    axes[1, 0].boxplot(data_for_box,
                      labels=['Dice', 'IoU', 'Precision', 'Recall'])
    axes[1, 0].set_title('Metrics Overview')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    # 5. Per-Class Dice Scores
    class_dice_data = []
    class_labels = []
    for label, scores in metrics['per_class'].items():
        if scores:
            class_dice_data.append(scores)
            class_labels.append(tumor_types[label])

    if class_dice_data:
        axes[1, 1].boxplot(class_dice_data, labels=class_labels)
        axes[1, 1].set_title('Dice Scores per Tumor Type')
        axes[1, 1].set_ylabel('Dice Score')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
        plt.setp(axes[1, 1].get_xticklabels(), rotation=45)

    # 6. Class Distribution Bar Chart
    class_counts = [actual_class_counts[i] for i in [1, 2, 3]]
    class_names = [tumor_types[i] for i in [1, 2, 3]]

    bars = axes[1, 2].bar(class_names, class_counts,
                         color=['skyblue', 'lightgreen', 'salmon'])
    axes[1, 2].set_title('Sample Distribution')
    axes[1, 2].set_ylabel('Number of Images')
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    for bar, count in zip(bars, class_counts):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       str(count), ha='center', va='bottom')

    plt.setp(axes[1, 2].get_xticklabels(), rotation=45)

    plt.suptitle(f'Brain Tumor Segmentation Results {"(WITH TTA)" if USE_TTA else "(NO TTA)"}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # ============================================
    # 10. SPEICHERE ERGEBNISSE
    # ============================================

    import json
    from datetime import datetime

    results_dict = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'evaluation_type': 'with_tta' if USE_TTA else 'no_tta',
        'tta_enabled': USE_TTA,
        'total_images_evaluated': len(metrics['dice']),
        'total_images_available': len(mat_files),
        'actual_class_distribution': actual_class_counts,
        'metrics': {
            'dice': {
                'mean': float(np.mean(metrics['dice'])),
                'std': float(np.std(metrics['dice'])),
                'min': float(np.min(metrics['dice'])),
                'max': float(np.max(metrics['dice'])),
                'median': float(np.median(metrics['dice']))
            },
            'iou': {
                'mean': float(np.mean(metrics['iou'])),
                'std': float(np.std(metrics['iou'])),
                'median': float(np.median(metrics['iou']))
            },
            'precision': {
                'mean': float(np.mean(metrics['precision'])),
                'std': float(np.std(metrics['precision']))
            },
            'recall': {
                'mean': float(np.mean(metrics['recall'])),
                'std': float(np.std(metrics['recall']))
            },
            'per_class': {}
        }
    }

    if USE_TTA and metrics['confidence']:
        results_dict['metrics']['tta_confidence'] = {
            'mean': float(np.mean(metrics['confidence'])),
            'std': float(np.std(metrics['confidence']))
        }

    # Per-class Metriken
    for label, scores in metrics['per_class'].items():
        if scores:
            results_dict['metrics']['per_class'][tumor_types[label]] = {
                'dice_mean': float(np.mean(scores)),
                'dice_std': float(np.std(scores)),
                'n_samples': len(scores)
            }

    # Makro-Durchschnitt
    class_means = [np.mean(scores) for scores in metrics['per_class'].values() if scores]
    if class_means:
        results_dict['metrics']['macro_dice'] = float(np.mean(class_means))

    if metrics['hausdorff']:
        results_dict['metrics']['hausdorff'] = {
            'mean': float(np.mean(metrics['hausdorff'])),
            'std': float(np.std(metrics['hausdorff'])),
            'median': float(np.median(metrics['hausdorff']))
        }

    # Speichere
    filename = f'/content/segmentation_results_{"tta" if USE_TTA else "no_tta"}.json'
    with open(filename, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n💾 Ergebnisse gespeichert in '{filename}'")

    # Download
    from google.colab import files
    files.download(filename)

    print("\n" + "="*70)
    print("✅ EVALUATION ERFOLGREICH ABGESCHLOSSEN!")
    print("="*70)
    print(f"📊 Evaluiert: {len(metrics['dice'])} Bilder")
    print(f"🎯 TTA: {'AKTIVIERT (6x Augmentationen)' if USE_TTA else 'DEAKTIVIERT'}")
    print(f"📈 Makro Dice Score: {np.mean(class_means):.4f}")
    print(f"📈 Standard Dice Score: {np.mean(metrics['dice']):.4f}")

    if USE_TTA:
        print(f"\n💡 TIPP: Setze USE_TTA = False (Zeile 642) um Geschwindigkeit zu vergleichen")
    else:
        print(f"\n💡 TIPP: Setze USE_TTA = True (Zeile 642) um Genauigkeit zu verbessern")

else:
    print("\n❌ Keine Bilder konnten evaluiert werden")
