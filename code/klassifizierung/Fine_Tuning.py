# ============================================
# BRAIN TUMOR CLASSIFICATION - FINE-TUNING
# Optimiert für Glioma vs Pituitary Verwechslung 
# ============================================

# Importieren der Bibliotheken 
from google.colab import drive
drive.mount('/content/drive')
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import glob
import re

print("="*80)
print("BRAIN TUMOR CLASSIFICATION - FINE-TUNING")
print("="*80)

# ============================================
# 1. KONFIGURATION
# ============================================

CONFIG = {
    'base_model': 'Devarshi/Brain_Tumor_Classification',
    'train_path': '/content/drive/My Drive/Training',
    'test_path': '/content/drive/My Drive/Testing_1/Testing',
    'output_dir': '/content/drive/My Drive/Brain_Tumor_FineTuned',
    'num_classes': 4,
    'class_names': ['glioma', 'meningioma', 'notumor', 'pituitary'],
    'batch_size': 16,
    'learning_rate': 2e-5,  # Sehr niedrig für Fine-Tuning
    'num_epochs': 5,
    'freeze_layers': True,  # Friere frühe Layer ein
    'num_layers_to_freeze': 2,  # Swin Transformer hat nur 4 Layer total!
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n Hardware: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ============================================
# 2. LABEL MAPPING & PARSING
# ============================================

LABEL_MAPPING = {
    'glioma': 0,
    'meningioma': 1,
    'notumor': 2,
    'pituitary': 3
}

def parse_tumor_filename(filename):
    """Extrahiert Label aus Dateinamen"""
    basename = os.path.basename(filename)
    pattern = r'(Te|Tr)-(gl|me|pi|no)_(\d+)'
    match = re.search(pattern, basename)

    if match:
        tumor_map = {
            'gl': 'glioma',
            'me': 'meningioma',
            'pi': 'pituitary',
            'no': 'notumor'
        }
        return tumor_map.get(match.group(2), None)

    # Fallback
    for key in ['glioma', 'meningioma', 'pituitary', 'notumor']:
        if key in basename.lower() or key.replace('tumor', '_tumor') in basename.lower():
            return key

    return None

# ============================================
# 3. DATASET KLASSE
# ============================================

class BrainTumorDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = []
        self.labels = []
        self.processor = processor

        # Filtere nur Bilder mit gültigen Labels
        for path in image_paths:
            label_name = parse_tumor_filename(path)
            if label_name and label_name in LABEL_MAPPING:
                self.image_paths.append(path)
                self.labels.append(LABEL_MAPPING[label_name])

        print(f"   Geladene Bilder: {len(self.image_paths)}")

        # Zeige Verteilung
        from collections import Counter
        label_counts = Counter(self.labels)
        for label_name, label_id in sorted(LABEL_MAPPING.items(), key=lambda x: x[1]):
            print(f"   {label_name:12}: {label_counts[label_id]:4} Bilder")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        # Preprocessing mit Hugging Face Processor
        encoding = self.processor(image, return_tensors='pt')
        encoding = {key: val.squeeze() for key, val in encoding.items()}
        encoding['labels'] = torch.tensor(label)

        return encoding

# ============================================
# 4. DATEN LADEN
# ============================================

print("\n Lade Daten...")

# Sammle alle Bilder
train_images = []
test_images = []

for ext in ('*.jpg', '*.jpeg', '*.png'):
    train_images.extend(glob.glob(os.path.join(CONFIG['train_path'], '**', ext), recursive=True))
    test_images.extend(glob.glob(os.path.join(CONFIG['test_path'], '**', ext), recursive=True))

print(f"\n📊 Gefundene Bilder:")
print(f"   Training:  {len(train_images)}")
print(f"   Test:      {len(test_images)}")

# ============================================
# 5. MODELL & PROCESSOR LADEN
# ============================================

print("\n Lade Modell und Processor...")

processor = AutoImageProcessor.from_pretrained(CONFIG['base_model'])
model = AutoModelForImageClassification.from_pretrained(
    CONFIG['base_model'],
    num_labels=CONFIG['num_classes'],
    ignore_mismatched_sizes=True
)

# ============================================
# 6. LAYER FREEZING (basierend auf der Inspektion Klassifikationsmodell_Inspektion.py)
# ============================================

if CONFIG['freeze_layers']:
    print(f"\n Friere erste {CONFIG['num_layers_to_freeze']} Layer ein...")

    # Swin Transformer Struktur
    if hasattr(model, 'swin'):
        base_model = model.swin

        # Friere Embeddings ein
        if hasattr(base_model, 'embeddings'):
            for param in base_model.embeddings.parameters():
                param.requires_grad = False
            print("   ✅ Embeddings eingefroren")

        # Friere frühe Encoder-Layer ein
        if hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layers'):
            layers = base_model.encoder.layers
            total_layers = len(layers)
            print(f"   📊 Gefunden: {total_layers} Encoder-Layer")

            for i, layer in enumerate(layers):
                if i < CONFIG['num_layers_to_freeze']:
                    for param in layer.parameters():
                        param.requires_grad = False
                    print(f"    Layer {i} eingefroren")
                else:
                    print(f"    Layer {i} trainierbar")

    # Klassifier bleibt immer trainierbar
    if hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
        print("    Classifier trainierbar")

    # Zeige trainierbare Parameter
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n    Trainierbare Parameter: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")

model.to(device)

# ============================================
# 7. DATASETS ERSTELLEN
# ============================================

print("\n Erstelle Datasets...")
print("Training Set:")
train_dataset = BrainTumorDataset(train_images, processor)

print("\nTest Set:")
test_dataset = BrainTumorDataset(test_images, processor)

# ============================================
# 8. METRIKEN BERECHNUNG
# ============================================

def compute_metrics(eval_pred):
    """Berechne Metriken während Training"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ============================================
# 9. TRAINING KONFIGURATION
# ============================================

training_args = TrainingArguments(
    output_dir=CONFIG['output_dir'],
    num_train_epochs=CONFIG['num_epochs'],
    per_device_train_batch_size=CONFIG['batch_size'],
    per_device_eval_batch_size=CONFIG['batch_size'],
    learning_rate=CONFIG['learning_rate'],
    weight_decay=0.01,

    # Evaluation
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',

    # Logging
    logging_dir=os.path.join(CONFIG['output_dir'], 'logs'),
    logging_steps=50,

    # Performance
    fp16=torch.cuda.is_available(),  # Mixed Precision auf GPU
    dataloader_num_workers=2,

    # Speichern
    save_total_limit=2,  # Nur beste 2 Checkpoints behalten
)

# ============================================
# 10. TRAINER ERSTELLEN & TRAINING STARTEN
# ============================================

print("\n🚀 Starte Fine-Tuning...")
print(f"   Epochs: {CONFIG['num_epochs']}")
print(f"   Batch Size: {CONFIG['batch_size']}")
print(f"   Learning Rate: {CONFIG['learning_rate']}")
print("="*80)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Training starten
train_result = trainer.train()

print("\n✅ Training abgeschlossen!")
print(f"   Final Loss: {train_result.training_loss:.4f}")

# ============================================
# 11. FINALE EVALUATION
# ============================================

print("\n Finale Evaluation auf Test-Set...")
eval_results = trainer.evaluate()

print("\n" + "="*80)
print(" FINALE METRIKEN")
print("="*80)
print(f"Accuracy:   {eval_results['eval_accuracy']:.4f}")
print(f"Precision:  {eval_results['eval_precision']:.4f}")
print(f"Recall:     {eval_results['eval_recall']:.4f}")
print(f"F1-Score:   {eval_results['eval_f1']:.4f}")

# ============================================
# 12. MODELL SPEICHERN
# ============================================

print("\n Speichere finales Modell...")
final_model_path = os.path.join(CONFIG['output_dir'], 'final_model')
trainer.save_model(final_model_path)
processor.save_pretrained(final_model_path)

print(f" Modell gespeichert: {final_model_path}")

# ============================================
# 13. VERGLEICH MIT ORIGINAL
# ============================================

print("\n" + "="*80)
print(" VERBESSERUNG")
print("="*80)
print(f"Original Modell:      91.0%")
print(f"Fine-Tuned Modell:    {eval_results['eval_accuracy']*100:.1f}%")
print(f"Verbesserung:         {(eval_results['eval_accuracy']-0.91)*100:+.1f}%")

print("\n NÄCHSTE SCHRITTE:")
print("1. Verwende das fine-tuned Modell in deinem Evaluations-Script")
print(f"2. Lade mit: AutoModelForImageClassification.from_pretrained('{final_model_path}')")
print("3. Teste besonders die Glioma vs Pituitary Konfusion")

print("\n" + "="*80)
print(" FINE-TUNING ABGESCHLOSSEN!")
print("="*80)
