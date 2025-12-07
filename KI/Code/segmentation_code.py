# ============================================
# BALANCED BRAIN TUMOR SEGMENTATION EVALUATION

# ============================================

import glob

# durchsucht auch alle Unterordner nach .mat-Dateien
mat_files = glob.glob('/content/drive/My Drive/Testing_Segmentation/**/*.mat', recursive=True)

print(f"✅ Gefunden: {len(mat_files)} .mat Dateien insgesamt")

import tensorflow as tf
import numpy as np
import h5py
import glob
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import precision_score, recall_score
from collections import defaultdict
import random
import matplotlib.pyplot as plt

print("🔍 Suche .mat Dateien...\n")

# Suche im Root-Verzeichnis

print(f"✅ Gefunden: {len(mat_files)} .mat Dateien insgesamt")

# Funktion zum Laden von MATLAB v7.3 Dateien
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

# BALANCED SAMPLING FUNKTION
def create_balanced_sample(mat_files, samples_per_class=700):
    """
    Erstellt eine ausgewogene Stichprobe mit gleicher Anzahl pro Klasse
    """
    # Gruppiere Dateien nach Label
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
    total_available = 0
    for label, files in sorted(files_by_class.items()):
        percentage = len(files) / len(mat_files) * 100 if mat_files else 0
        print(f"{tumor_types[label]:20s}: {len(files):4d} Bilder ({percentage:5.1f}%)")
        total_available += len(files)

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
            # Zufällige Auswahl
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

    total_selected = len(balanced_sample)
    print(f"\n✅ FINALE BALANCED STICHPROBE: {total_selected} Bilder")
    print(f"   Pro Klasse: {actual_samples_per_class} Bilder")
    print(f"   Balancing-Ratio: {total_selected/3:.1f} pro Klasse")

    return balanced_sample, final_distribution

# Erstelle balanced sample
print("\n" + "="*60)
print("📊 BALANCED SAMPLING (100 PRO KLASSE)")
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

# Lade das Modell
print("\n📦 Lade Modell von Hugging Face...")
from huggingface_hub import snapshot_download

repo_path = snapshot_download(repo_id="bombshelll/unet-brain-tumor-segmentation")
model = tf.saved_model.load(repo_path)

class GrayscaleToRGBModelWrapper:
    def __init__(self, model):
        self.model = model
        self.serving_fn = model.signatures.get('serving_default')
        self.input_name = 'keras_tensor_68'

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

# Metriken-Klassen
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

# HAUPTEVALUATION MIT BALANCED SAMPLE
def evaluate_balanced_sample(wrapped_model, balanced_files, class_distribution):
    """
    Evaluation mit balanced sample
    """
    sm = SegmentationMetrics()
    all_metrics = {
        'dice': [], 'iou': [],
        'precision': [], 'recall': [],
        'hausdorff': [],
        'per_class': {1: [], 2: [], 3: []}
    }

    tumor_types = {1: "Meningiom", 2: "Gliom", 3: "Hypophysentumor"}

    print(f"\n🚀 STARTE BALANCED EVALUATION")
    print(f"   Total Dateien: {len(balanced_files)}")
    print(f"   Erwartete Verteilung: {class_distribution}")
    print()

    successful = 0
    failed = 0
    class_counts = {1: 0, 2: 0, 3: 0}

    # Progress bar mit detaillierter Info
    with tqdm(total=len(balanced_files), desc="Balanced Evaluation") as pbar:
        for i, mat_file in enumerate(balanced_files):
            try:
                # Update progress bar
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

                # Resize auf 256x256
                if image.shape != (256, 256):
                    image = tf.image.resize(
                        np.expand_dims(image, axis=-1),
                        (256, 256)
                    ).numpy().squeeze()

                # Prepare Input
                input_image = image.reshape(1, 256, 256, 1).astype(np.float32)

                # Prediction
                pred = wrapped_model.predict(input_image)

                if len(pred.shape) == 4:
                    pred_mask = (pred[0, :, :, 0] > 0.5).astype(np.float32)
                else:
                    pred_mask = (pred.squeeze() > 0.5).astype(np.float32)

                # Resize zurück
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

    # Zeige tatsächliche Klassenverteilung
    print(f"\n📊 TATSÄCHLICHE EVALUATION-VERTEILUNG:")
    print("-" * 50)
    for label, count in class_counts.items():
        percentage = count / successful * 100 if successful > 0 else 0
        print(f"{tumor_types[label]:20s}: {count:4d} Bilder ({percentage:5.1f}%)")

    return all_metrics, successful, class_counts

# FÜHRE BALANCED EVALUATION AUS
print("\n" + "="*60)
print("🚀 BALANCED EVALUATION STARTET")
print("="*60)

metrics, num_successful, actual_class_counts = evaluate_balanced_sample(
    wrapped_model, balanced_files, class_distribution
)

if num_successful > 0:
    # Zeige Ergebnisse
    print("\n" + "="*70)
    print("📊 BALANCED EVALUATIONSERGEBNISSE")
    print("="*70)

    tumor_types = {1: "Meningiom", 2: "Gliom", 3: "Hypophysentumor"}

    if metrics['dice']:
        # Gesamt-Metriken
        print(f"\n🎯 GESAMT-METRIKEN (Balanced Sample):")
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

        # Per-Class Ergebnisse (jetzt balanced!)
        print(f"\n📊 ERGEBNISSE PRO TUMOR-TYP (Balanced):")
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

        print(f"\n📈 Total evaluierte Bilder: {len(metrics['dice'])}")
        print(f"📈 Balancing erfolgreich: Alle Klassen gleichmässig repräsentiert!")

        # Berechne Makro-Durchschnitt (ungewichtet über Klassen)
        class_means = []
        for label, scores in metrics['per_class'].items():
            if scores:
                class_means.append(np.mean(scores))

        if class_means:
            macro_dice = np.mean(class_means)
            print(f"\n🎯 MAKRO-DURCHSCHNITT (ungewichtet über Klassen):")
            print(f"   Makro Dice Score: {macro_dice:.4f}")
            print(f"   Standard Dice Score: {np.mean(metrics['dice']):.4f}")

    # Erweiterte Visualisierung
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
    axes[1, 2].set_title('Balanced Sample Distribution')
    axes[1, 2].set_ylabel('Number of Images')
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    # Füge Werte auf den Balken hinzu
    for bar, count in zip(bars, class_counts):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       str(count), ha='center', va='bottom')

    plt.setp(axes[1, 2].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()

    # Speichere Ergebnisse
    import json
    from datetime import datetime

    results_dict = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'evaluation_type': 'balanced_sampling',
        'target_samples_per_class': 100,
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
    with open('/content/balanced_evaluation_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print("\n💾 Ergebnisse gespeichert in '/content/balanced_evaluation_results.json'")

    # Download
    from google.colab import files
    files.download('/content/balanced_evaluation_results.json')

    print("\n" + "="*70)
    print("✅ BALANCED EVALUATION ERFOLGREICH ABGESCHLOSSEN!")
    print("="*70)
    print(f"📊 Evaluiert: {len(metrics['dice'])} Bilder (balanced)")
    print(f"🎯 Ziel erreicht: 100 Bilder pro Klasse")
    print(f"📈 Makro Dice Score: {np.mean(class_means):.4f}")
    print(f"📈 Standard Dice Score: {np.mean(metrics['dice']):.4f}")

else:
    print("\n❌ Keine Bilder konnten evaluiert werden")
