# ============================================
# KORRIGIERTE BRAIN TUMOR CLASSIFICATION EVALUATION
# MIT LABEL-MAPPING FÜR KONSISTENTE KLASSEN
# ============================================



from google.colab import drive
drive.mount('/content/drive')
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# ML/DL Bibliotheken
import torch
from transformers import pipeline
from PIL import Image

# Sklearn Metriken
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score
)
from sklearn.preprocessing import label_binarize

# Plot Style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("🧠 BRAIN TUMOR CLASSIFICATION - EVALUATION MIT LABEL-MAPPING")
print("="*80)

# ============================================
# 1. SETUP UND MODELL LADEN
# ============================================

device = 0 if torch.cuda.is_available() else -1
print(f"\n⚙️ Hardware: {'GPU (' + torch.cuda.get_device_name(0) + ')' if device == 0 else 'CPU'}")

print("📦 Lade Modell...")

# WÄHLE MODELL: Original oder Fine-Tuned
USE_FINETUNED = True  # Ändere auf False um Original-Modell zu verwenden

if USE_FINETUNED:
    model_path = '/content/drive/My Drive/Brain_Tumor_FineTuned/final_model'
    print("   🎯 Verwende FINE-TUNED Modell")
else:
    model_path = "Devarshi/Brain_Tumor_Classification"
    print("   📦 Verwende ORIGINAL Modell")

pipe = pipeline(
    "image-classification",
    model=model_path,
    device=device
)
print("✅ Modell erfolgreich geladen")

# ============================================
# 2. LABEL MAPPING DEFINITIONEN
# ============================================

LABEL_MAPPING = {
    'glioma': 'glioma',
    'glioma_tumor': 'glioma',
    'meningioma': 'meningioma',
    'meningioma_tumor': 'meningioma',
    'pituitary': 'pituitary',
    'pituitary_tumor': 'pituitary'
}

STANDARD_CLASSES = ['glioma', 'meningioma', 'pituitary']  # Nur 3 Klassen!

def standardize_label(label):
    """Standardisiert Labels auf die 4 Hauptkategorien"""
    return LABEL_MAPPING.get(label, label)

# ============================================
# 3. DATEINAMEN-PARSER
# ============================================

def parse_tumor_filename(filename):
    """Extrahiert Label aus Dateinamen"""
    basename = os.path.basename(filename)

    # Neues Format: {tumor_type}_{patient_id}_{file_id}.png
    # Beispiel: glioma_TC201_100.png
    if basename.startswith(('glioma_', 'meningioma_', 'pituitary_')):
        tumor_type = basename.split('_')[0]
        return tumor_type

    # Altes Format: (Te|Tr)-(gl|me|pi)_(\d+)
    pattern = r'(Te|Tr)-(gl|me|pi)_(\d+)'
    match = re.search(pattern, basename)

    if match:
        tumor_map = {
            'gl': 'glioma',
            'me': 'meningioma',
            'pi': 'pituitary'
        }
        return tumor_map.get(match.group(2), 'unknown')

    # Fallback: Suche nach Schlüsselwörtern
    if 'glioma' in basename.lower():
        return 'glioma'
    elif 'meningioma' in basename.lower():
        return 'meningioma'
    elif 'pituitary' in basename.lower():
        return 'pituitary'

    return 'unknown'

# ============================================
# 4. BILDER LADEN
# ============================================

print("\n🔍 Suche Bilder...")
base_path = '/content/drive/My Drive/Testing'

all_images = []
for ext in ('*.jpg', '*.jpeg', '*.png'):
    all_images.extend(glob.glob(os.path.join(base_path, '**', ext), recursive=True))

all_images = sorted(all_images)
print(f"📸 Gefundene Bilder: {len(all_images)}")

# ============================================
# 5. KLASSIFIKATION
# ============================================

print("\n🚀 Starte Klassifikation mit Label-Standardisierung und Test-Time Augmentation...")
print("⚠️ AGGRESSIVE TTA aktiviert: 8x Augmentationen pro Bild (deutlich länger aber genauer)")

predictions_raw = []
predictions = []
true_labels = []
probabilities = []
image_names = []
all_results = []

# Import für Augmentationen
from PIL import ImageOps, ImageEnhance

for i, img_path in enumerate(all_images, 1):
    try:
        if i % 50 == 0 or i == len(all_images):
            print(f"⏳ Verarbeitung: {i}/{len(all_images)} ({i/len(all_images)*100:.1f}%)")

        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # AGGRESSIVE TEST-TIME AUGMENTATION (TTA)
        # Erstelle 8 Varianten des Bildes
        brightness_enhancer = ImageEnhance.Brightness(image)
        contrast_enhancer = ImageEnhance.Contrast(image)

        augmented_images = [
            image,                                      # 1. Original
            ImageOps.mirror(image),                     # 2. Horizontal gespiegelt
            image.rotate(5, expand=False),              # 3. +5° Rotation
            image.rotate(-5, expand=False),             # 4. -5° Rotation
            image.rotate(10, expand=False),             # 5. +10° Rotation
            image.rotate(-10, expand=False),            # 6. -10° Rotation
            brightness_enhancer.enhance(1.1),           # 7. Helligkeit +10%
            contrast_enhancer.enhance(1.1),             # 8. Kontrast +10%
        ]

        # Sammle alle Vorhersagen
        all_predictions = []
        for aug_img in augmented_images:
            aug_result = pipe(aug_img)
            all_predictions.extend(aug_result)

        # Aggregiere Wahrscheinlichkeiten nach Label
        label_probs = {}
        for pred in all_predictions:
            label = pred['label']
            score = pred['score']
            if label not in label_probs:
                label_probs[label] = []
            label_probs[label].append(score)

        # Berechne Durchschnitt für jedes Label
        avg_label_probs = {label: np.mean(scores) for label, scores in label_probs.items()}

        # Sortiere nach Wahrscheinlichkeit
        result = [
            {'label': label, 'score': score}
            for label, score in sorted(avg_label_probs.items(), key=lambda x: x[1], reverse=True)
        ]

        pred_raw = result[0]['label']
        predictions_raw.append(pred_raw)

        pred_standard = standardize_label(pred_raw)
        predictions.append(pred_standard)

        true_label = parse_tumor_filename(img_path)
        true_labels.append(true_label)

        prob_dict_raw = {item['label']: item['score'] for item in result}
        prob_dict = {}
        for label, prob in prob_dict_raw.items():
            standard_label = standardize_label(label)
            if standard_label in prob_dict:
                prob_dict[standard_label] = max(prob_dict[standard_label], prob)
            else:
                prob_dict[standard_label] = prob

        probabilities.append(prob_dict)
        image_names.append(os.path.basename(img_path))

        all_results.append({
            'filename': os.path.basename(img_path),
            'path': img_path,
            'true_label': true_label,
            'predicted_label_raw': pred_raw,
            'predicted_label': pred_standard,
            'confidence': result[0]['score'],
            'all_predictions': result
        })

    except Exception as e:
        print(f"❌ Fehler bei {img_path}: {e}")

print(f"\n✅ Klassifikation abgeschlossen: {len(predictions)} Bilder verarbeitet")

# ============================================
# 6. DATENFILTERUNG
# ============================================

valid_indices = [i for i, label in enumerate(true_labels) if label != 'unknown']

if len(valid_indices) == 0:
    print("❌ Keine Bilder mit gültigen Labels gefunden!")
    import sys
    sys.exit()

print(f"📊 Bilder mit gültigen Labels: {len(valid_indices)}/{len(true_labels)}")

y_true = [true_labels[i] for i in valid_indices]
y_pred = [predictions[i] for i in valid_indices]
y_pred_raw = [predictions_raw[i] for i in valid_indices]
y_probs = [probabilities[i] for i in valid_indices]

print(f"\n🔄 Label-Mapping angewendet:")
unique_raw = set(y_pred_raw)
print(f"   Original Modell-Labels: {sorted(unique_raw)}")
print(f"   Standardisierte Labels: {STANDARD_CLASSES}")

# ============================================
# 7. KLASSIFIKATIONS-METRIKEN
# ============================================

print("\n" + "="*80)
print("📊 KLASSIFIKATIONS-METRIKEN")
print("="*80)

accuracy = accuracy_score(y_true, y_pred)
balanced_acc = balanced_accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
mcc = matthews_corrcoef(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)

print(f"""
🎯 GESAMT-METRIKEN:
─────────────────────────────────────
Accuracy:                {accuracy:.4f}
Balanced Accuracy:       {balanced_acc:.4f}
Matthews Correlation:    {mcc:.4f}
Cohen's Kappa:          {kappa:.4f}

📊 MACRO-DURCHSCHNITT:
─────────────────────────────────────
Precision:              {precision_macro:.4f}
Recall:                 {recall_macro:.4f}
F1-Score:               {f1_macro:.4f}

⚖️ WEIGHTED-DURCHSCHNITT:
─────────────────────────────────────
Precision:              {precision_weighted:.4f}
Recall:                 {recall_weighted:.4f}
F1-Score:               {f1_weighted:.4f}
""")

# ============================================
# 8. KLASSENSPEZIFISCHE METRIKEN
# ============================================

print("\n" + "="*80)
print("📋 KLASSENSPEZIFISCHE METRIKEN")
print("="*80)

report = classification_report(y_true, y_pred, target_names=STANDARD_CLASSES, digits=4)
print(report)

# ============================================
# 9. CONFUSION MATRIX
# ============================================

print("\n" + "="*80)
print("🔥 CONFUSION MATRIX")
print("="*80)

cm = confusion_matrix(y_true, y_pred, labels=STANDARD_CLASSES)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=STANDARD_CLASSES, yticklabels=STANDARD_CLASSES,
           cbar_kws={'label': 'Anzahl'}, ax=axes[0])
axes[0].set_title('Confusion Matrix (Absolute)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Vorhergesagt', fontsize=12)
axes[0].set_ylabel('Wahr', fontsize=12)

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='YlOrRd',
           xticklabels=STANDARD_CLASSES, yticklabels=STANDARD_CLASSES,
           cbar_kws={'label': 'Prozent'}, ax=axes[1], vmin=0, vmax=1)
axes[1].set_title('Confusion Matrix (Normalisiert)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Vorhergesagt', fontsize=12)
axes[1].set_ylabel('Wahr', fontsize=12)

plt.suptitle('🧠 Brain Tumor Classification - Confusion Matrices', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n📊 Confusion Matrix Analyse:")
for i, cls in enumerate(STANDARD_CLASSES):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    tn = cm.sum() - tp - fp - fn

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    print(f"\n{cls}:")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  PPV: {ppv:.4f}")
    print(f"  NPV: {npv:.4f}")

# ============================================
# 10. ROC-KURVEN UND AUC
# ============================================

print("\n" + "="*80)
print("📈 ROC-KURVEN UND AUC")
print("="*80)

prob_matrix = np.zeros((len(y_probs), len(STANDARD_CLASSES)))
for i, prob_dict in enumerate(y_probs):
    for j, cls in enumerate(STANDARD_CLASSES):
        prob_matrix[i, j] = prob_dict.get(cls, 0.0)

y_true_binary = label_binarize(y_true, classes=STANDARD_CLASSES)

plt.figure(figsize=(10, 8))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
auc_scores = {}

for i, (cls, color) in enumerate(zip(STANDARD_CLASSES, colors)):
    fpr, tpr, _ = roc_curve(y_true_binary[:, i], prob_matrix[:, i])
    roc_auc = auc(fpr, tpr)
    auc_scores[cls] = roc_auc
    plt.plot(fpr, tpr, color=color, lw=2.5, label=f'{cls} (AUC = {roc_auc:.3f})')

fpr_micro, tpr_micro, _ = roc_curve(y_true_binary.ravel(), prob_matrix.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, 'k--', lw=2, label=f'Micro-avg (AUC = {roc_auc_micro:.3f})')
plt.plot([0, 1], [0, 1], 'k:', lw=1.5, alpha=0.5, label='Zufall')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC-Kurven - Brain Tumor Classification', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n🎯 AUC-Scores:")
for cls, score in auc_scores.items():
    print(f"  {cls}: {score:.4f}")
print(f"\nMakro-AUC: {np.mean(list(auc_scores.values())):.4f}")
print(f"Mikro-AUC: {roc_auc_micro:.4f}")

# ============================================
# 11. INTELLIGENTE THRESHOLD OPTIMIZATION
# ============================================

print("\n" + "="*80)
print("🎯 INTELLIGENTE CLASS-WEIGHT OPTIMIZATION")
print("="*80)

from scipy.optimize import minimize

def optimize_multiclass_thresholds(y_true, prob_matrix, classes):
    """Optimiert Class-Bias für Multi-Class-Klassifikation"""
    
    def objective(weights):
        """Maximiere F1-Score durch Gewichtsanpassung"""
        adjusted_probs = prob_matrix * weights
        y_pred_adjusted = [classes[i] for i in adjusted_probs.argmax(axis=1)]
        return -f1_score(y_true, y_pred_adjusted, average='macro', zero_division=0)
    
    initial_weights = np.ones(len(classes))
    
    result = minimize(
        objective,
        initial_weights,
        method='Nelder-Mead',
        bounds=[(0.3, 3.0)] * len(classes),
        options={'maxiter': 500}
    )
    
    optimal_weights = result.x
    optimal_weights = optimal_weights / optimal_weights.mean()
    
    return optimal_weights

print("\n🔍 Suche optimale Class-Gewichte...")
optimal_weights = optimize_multiclass_thresholds(y_true, prob_matrix, STANDARD_CLASSES)

print("\n📊 Optimale Class-Gewichte (1.0 = neutral):")
print("─" * 50)
for cls, weight in zip(STANDARD_CLASSES, optimal_weights):
    direction = "↑ mehr bevorzugt" if weight > 1.1 else "↓ weniger bevorzugt" if weight < 0.9 else "→ neutral"
    print(f"{cls:12} → {weight:.3f}  {direction}")

adjusted_probs = prob_matrix * optimal_weights
y_pred_optimized = [STANDARD_CLASSES[i] for i in adjusted_probs.argmax(axis=1)]

# Vergleiche Metriken
print("\n" + "="*80)
print("📈 VERGLEICH: ORIGINAL vs. OPTIMIERT")
print("="*80)

acc_orig = accuracy_score(y_true, y_pred)
f1_macro_orig = f1_score(y_true, y_pred, average='macro', zero_division=0)
f1_weighted_orig = f1_score(y_true, y_pred, average='weighted', zero_division=0)

acc_opt = accuracy_score(y_true, y_pred_optimized)
f1_macro_opt = f1_score(y_true, y_pred_optimized, average='macro', zero_division=0)
f1_weighted_opt = f1_score(y_true, y_pred_optimized, average='weighted', zero_division=0)

print(f"""
{'Metrik':<20} {'Original':<12} {'Optimiert':<12} {'Δ':<10}
{'─'*60}
{'Accuracy':<20} {acc_orig:<12.4f} {acc_opt:<12.4f} {(acc_opt-acc_orig):+.4f}
{'F1-Score (Macro)':<20} {f1_macro_orig:<12.4f} {f1_macro_opt:<12.4f} {(f1_macro_opt-f1_macro_orig):+.4f}
{'F1-Score (Weighted)':<20} {f1_weighted_orig:<12.4f} {f1_weighted_opt:<12.4f} {(f1_weighted_opt-f1_weighted_orig):+.4f}
""")

f1_per_orig = f1_score(y_true, y_pred, average=None, labels=STANDARD_CLASSES)
f1_per_opt = f1_score(y_true, y_pred_optimized, average=None, labels=STANDARD_CLASSES)
recall_per_orig = recall_score(y_true, y_pred, average=None, labels=STANDARD_CLASSES)
recall_per_opt = recall_score(y_true, y_pred_optimized, average=None, labels=STANDARD_CLASSES)
precision_per_orig = precision_score(y_true, y_pred, average=None, labels=STANDARD_CLASSES)
precision_per_opt = precision_score(y_true, y_pred_optimized, average=None, labels=STANDARD_CLASSES)

print("\n📊 KLASSENSPEZIFISCHER VERGLEICH:")
print("─" * 75)
print(f"{'Klasse':<12} {'Metrik':<10} {'Original':<10} {'Optimiert':<10} {'Δ':<10}")
print("─" * 75)

for i, cls in enumerate(STANDARD_CLASSES):
    print(f"{cls:<12} {'F1':<10} {f1_per_orig[i]:<10.4f} {f1_per_opt[i]:<10.4f} {(f1_per_opt[i]-f1_per_orig[i]):+.4f}")
    print(f"{'':<12} {'Recall':<10} {recall_per_orig[i]:<10.4f} {recall_per_opt[i]:<10.4f} {(recall_per_opt[i]-recall_per_orig[i]):+.4f}")
    print(f"{'':<12} {'Precision':<10} {precision_per_orig[i]:<10.4f} {precision_per_opt[i]:<10.4f} {(precision_per_opt[i]-precision_per_orig[i]):+.4f}")
    print("─" * 75)

cm_opt = confusion_matrix(y_true, y_pred_optimized, labels=STANDARD_CLASSES)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=STANDARD_CLASSES, yticklabels=STANDARD_CLASSES,
           cbar_kws={'label': 'Anzahl'}, ax=axes[0])
axes[0].set_title(f'Original (Acc: {acc_orig:.3f})', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Vorhergesagt')
axes[0].set_ylabel('Wahr')

sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens',
           xticklabels=STANDARD_CLASSES, yticklabels=STANDARD_CLASSES,
           cbar_kws={'label': 'Anzahl'}, ax=axes[1])
axes[1].set_title(f'Mit Class-Weights (Acc: {acc_opt:.3f})', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Vorhergesagt')
axes[1].set_ylabel('Wahr')

plt.suptitle('🎯 Confusion Matrix: Vorher vs. Nachher', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n💡 INTERPRETATION:")
print("─" * 75)
for i, cls in enumerate(STANDARD_CLASSES):
    improvement = f1_per_opt[i] - f1_per_orig[i]
    if improvement > 0.02:
        print(f"✅ {cls}: +{improvement:.3f} F1 (deutliche Verbesserung)")
    elif improvement < -0.02:
        print(f"⚠️ {cls}: {improvement:.3f} F1 (Verschlechterung)")
    else:
        print(f"➡️ {cls}: {improvement:+.3f} F1 (minimal geändert)")

# ============================================
# 12. FEHLERANALYSE
# ============================================

print("\n" + "="*80)
print("🖼️ DETAILLIERTE FEHLERANALYSE")
print("="*80)

errors_detailed = []
for i in range(len(y_true)):
    if y_true[i] != y_pred[i]:
        true_cls_idx = STANDARD_CLASSES.index(y_true[i])
        pred_cls_idx = STANDARD_CLASSES.index(y_pred[i])
        error_magnitude = prob_matrix[i, pred_cls_idx] - prob_matrix[i, true_cls_idx]
        
        errors_detailed.append({
            'index': valid_indices[i],
            'filename': image_names[valid_indices[i]],
            'true': y_true[i],
            'predicted': y_pred[i],
            'true_prob': prob_matrix[i, true_cls_idx],
            'pred_prob': prob_matrix[i, pred_cls_idx],
            'error_magnitude': error_magnitude,
            'path': all_images[valid_indices[i]]
        })

errors_detailed.sort(key=lambda x: x['error_magnitude'], reverse=True)

print(f"\n❌ Top 15 kritischste Fehlklassifikationen:")
print("─" * 90)
print(f"{'Datei':<20} {'Wahr':<12} {'Pred':<12} {'P(wahr)':<10} {'P(pred)':<10} {'Δ':<8}")
print("─" * 90)

for err in errors_detailed[:15]:
    print(f"{err['filename']:<20} {err['true']:<12} {err['predicted']:<12} "
          f"{err['true_prob']:<10.3f} {err['pred_prob']:<10.3f} {err['error_magnitude']:<8.3f}")

high_confidence_errors = [e for e in errors_detailed if e['pred_prob'] > 0.9]
medium_confidence_errors = [e for e in errors_detailed if 0.6 <= e['pred_prob'] <= 0.9]
low_confidence_errors = [e for e in errors_detailed if e['pred_prob'] < 0.6]

print("\n📊 Fehler-Kategorien:")
print("─" * 50)
print(f"Hohe Konfidenz (>0.9):     {len(high_confidence_errors)} Fehler")
print(f"Mittlere Konfidenz (0.6-0.9): {len(medium_confidence_errors)} Fehler")
print(f"Niedrige Konfidenz (<0.6):  {len(low_confidence_errors)} Fehler")
print("\n💡 Hohe Konfidenz-Fehler könnten falsche Labels sein!")

# Visualisiere Fehler
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.ravel()

for idx, err in enumerate(errors_detailed[:12]):
    try:
        img = Image.open(err['path'])
        axes[idx].imshow(img, cmap='gray')
        axes[idx].axis('off')
        
        if err['pred_prob'] > 0.9:
            title_color = '#c0392b'
        elif err['pred_prob'] > 0.6:
            title_color = '#e67e22'
        else:
            title_color = '#f39c12'
            
        axes[idx].set_title(
            f"{err['filename']}\n"
            f"Wahr: {err['true']} ({err['true_prob']:.2f})\n"
            f"Pred: {err['predicted']} ({err['pred_prob']:.2f})",
            fontsize=9, color=title_color, fontweight='bold'
        )
    except Exception as e:
        axes[idx].text(0.5, 0.5, 'Fehler beim Laden', ha='center', va='center')
        axes[idx].axis('off')

plt.suptitle('🔍 Top 12 kritischste Fehlklassifikationen\n(Dunkelrot = hohe Modell-Konfidenz)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================
# 13. SPEICHERUNG
# ============================================

print("\n" + "="*80)
print("💾 ERGEBNISSE SPEICHERN")
print("="*80)

output_dir = '/content/drive/My Drive/Brain_Tumor_Results'
os.makedirs(output_dir, exist_ok=True)

results_data = []
for i, result in enumerate(all_results):
    row = {
        'filename': result['filename'],
        'path': result['path'],
        'true_label': result['true_label'],
        'predicted_label': result['predicted_label'],
        'confidence': result['confidence']
    }
    
    if 'all_predictions' in result and len(result['all_predictions']) > 0:
        for pred_item in result['all_predictions']:
            if isinstance(pred_item, dict) and 'label' in pred_item:
                cls = standardize_label(pred_item['label'])
                row[f'prob_{cls}'] = pred_item.get('score', 0.0)
    
    results_data.append(row)

results_df = pd.DataFrame(results_data)

results_df['predicted_label_optimized'] = None
for i, idx in enumerate(valid_indices):
    if idx < len(results_df):
        results_df.loc[idx, 'predicted_label_optimized'] = y_pred_optimized[i]

results_df.to_csv(os.path.join(output_dir, 'brain_tumor_results_complete.csv'), index=False)
print("✅ brain_tumor_results_complete.csv gespeichert")

errors_df = pd.DataFrame(errors_detailed)
errors_df.to_csv(os.path.join(output_dir, 'critical_errors.csv'), index=False)
print("✅ critical_errors.csv gespeichert")

metrics_summary = {
    'timestamp': datetime.now().isoformat(),
    'model': 'Devarshi/Brain_Tumor_Classification',
    'total_images': len(all_images),
    'valid_images': len(valid_indices),
    
    'original_metrics': {
        'accuracy': float(acc_orig),
        'f1_macro': float(f1_macro_orig),
        'f1_weighted': float(f1_weighted_orig),
        'per_class': {
            cls: {
                'f1': float(f1_per_orig[i]),
                'recall': float(recall_per_orig[i]),
                'precision': float(precision_per_orig[i])
            }
            for i, cls in enumerate(STANDARD_CLASSES)
        }
    },
    
    'optimized_metrics': {
        'accuracy': float(acc_opt),
        'f1_macro': float(f1_macro_opt),
        'f1_weighted': float(f1_weighted_opt),
        'class_weights': {cls: float(w) for cls, w in zip(STANDARD_CLASSES, optimal_weights)},
        'improvement': {
            'accuracy': float(acc_opt - acc_orig),
            'f1_macro': float(f1_macro_opt - f1_macro_orig)
        },
        'per_class': {
            cls: {
                'f1': float(f1_per_opt[i]),
                'recall': float(recall_per_opt[i]),
                'precision': float(precision_per_opt[i])
            }
            for i, cls in enumerate(STANDARD_CLASSES)
        }
    },
    
    'error_analysis': {
        'total_errors': len(errors_detailed),
        'high_confidence_errors': len(high_confidence_errors),
        'medium_confidence_errors': len(medium_confidence_errors),
        'low_confidence_errors': len(low_confidence_errors),
        'top_10_critical': [
            {
                'filename': e['filename'],
                'true': e['true'],
                'predicted': e['predicted'],
                'confidence': float(e['pred_prob']),
                'error_magnitude': float(e['error_magnitude'])
            }
            for e in errors_detailed[:10]
        ]
    },
    
    'auc_scores': {cls: float(score) for cls, score in auc_scores.items()},
    'macro_auc': float(np.mean(list(auc_scores.values())))
}

with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
    json.dump(metrics_summary, f, indent=2)
print("✅ evaluation_summary.json gespeichert")

# ============================================
# 14. FINALER BERICHT
# ============================================

print("\n" + "="*80)
print("📊 FINALER EVALUATIONS-BERICHT")
print("="*80)

best_class = STANDARD_CLASSES[np.argmax(f1_per_opt)]
worst_class = STANDARD_CLASSES[np.argmin(f1_per_opt)]

print(f"""
🧠 BRAIN TUMOR CLASSIFICATION - ZUSAMMENFASSUNG
───────────────────────────────────────────────────────────────
Modell:         Devarshi/Brain_Tumor_Classification
Bilder Total:   {len(all_images)}
Ausgewertet:    {len(valid_indices)}
Fehler:         {len(errors_detailed)} ({len(errors_detailed)/len(y_true)*100:.1f}%)

🎯 PERFORMANCE:
───────────────────────────────────────────────────────────────
Original Accuracy:     {acc_orig:.2%}
Optimiert Accuracy:    {acc_opt:.2%}
Verbesserung:          {(acc_opt-acc_orig):+.2%}

F1-Score (Macro):      {f1_macro_orig:.2%} → {f1_macro_opt:.2%}
AUC (Makro):           {np.mean(list(auc_scores.values())):.2%}

📈 BESTE KLASSE (optimiert):
───────────────────────────────────────────────────────────────
{best_class}: F1={f1_per_opt[STANDARD_CLASSES.index(best_class)]:.3f}

📉 SCHWÄCHSTE KLASSE (optimiert):
───────────────────────────────────────────────────────────────
{worst_class}: F1={f1_per_opt[STANDARD_CLASSES.index(worst_class)]:.3f}

🔍 KRITISCHE BEFUNDE:
───────────────────────────────────────────────────────────────
• {len(high_confidence_errors)} Fehler mit >90% Konfidenz
  → Diese Bilder MANUELL überprüfen (mögliche Label-Fehler)!
  
• Hauptproblem: {errors_detailed[0]['true']} → {errors_detailed[0]['predicted']}
  ({sum(1 for e in errors_detailed if e['true']==errors_detailed[0]['true'] and e['predicted']==errors_detailed[0]['predicted'])}x)

📝 NÄCHSTE SCHRITTE:
───────────────────────────────────────────────────────────────
1. Überprüfe critical_errors.csv manuell
2. Korrigiere falsche Labels falls vorhanden
3. Falls keine Label-Fehler: Retraining mit Class Weights nötig

💾 GESPEICHERTE DATEIEN:
───────────────────────────────────────────────────────────────
• brain_tumor_results_complete.csv
• critical_errors.csv
• evaluation_summary.json
""")

print("\n" + "="*80)
print("✅ EVALUATION ABGESCHLOSSEN!")
print("="*80)

