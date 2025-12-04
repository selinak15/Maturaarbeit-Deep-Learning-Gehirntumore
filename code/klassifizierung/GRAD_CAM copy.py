import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ============================================================
# 1) Hardware & Pfade
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n Hardware: {device}")

model_path = "/content/drive/My Drive/Brain_Tumor_FineTuned/final_model" #Fine Tuned Modell
image_folder = "/content/drive/MyDrive/Testing_Segmentation/images"

# Optional: Nur bestimmte Klassen visualisieren (None = alle)
FILTER_CLASSES = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']  # oder None für alle

# ============================================================
# 2) Fine-Tuned Modell laden
# ============================================================
processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path).to(device)
model.eval()
labels = model.config.id2label
num_classes = len(labels)
print("✅ Fine-Tuned Modell geladen")
print(f"Labels: {labels}")
print(f"Anzahl Klassen: {num_classes}")

# ============================================================
# 3) Alle Bilder im Ordner sammeln
# ============================================================
all_images = []

for root, dirs, files in os.walk(image_folder):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            filepath = os.path.join(root, f)
            all_images.append(filepath)

print(f"Anzahl Bilder gefunden: {len(all_images)}")

if len(all_images) == 0:
    print(" Keine Bilder gefunden!")
    exit()

# ============================================================
# 4) Predictions für alle Bilder berechnen
# ============================================================
print("\n🔄 Berechne Predictions für alle Bilder...")
pred_logits = []
pred_classes = []

for idx, img_path in enumerate(all_images):
    if idx % 100 == 0:
        print(f"  Verarbeite Bild {idx+1}/{len(all_images)}")

    try:
        img = Image.open(img_path).convert("RGB")
        inputs = processor(img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]
            pred_logits.append(logits)
            pred_classes.append(np.argmax(logits))
    except Exception as e:
        print(f"   Fehler bei {img_path}: {e}")
        continue

pred_logits = np.array(pred_logits)
pred_classes = np.array(pred_classes)
print("✅ Predictions berechnet")

# Statistik anzeigen
print("\n Prediction-Verteilung:")
for class_idx in range(num_classes):
    count = np.sum(pred_classes == class_idx)
    percentage = count/len(all_images)*100 if len(all_images) > 0 else 0
    print(f"  {labels[class_idx]}: {count} Bilder ({percentage:.1f}%)")

# ============================================================
# 5) Top-5 Bilder pro vorhergesagter Klasse auswählen
# ============================================================
num_top = 5
top_images_per_class = {}

print("\n Wähle Top-5 Bilder pro Klasse aus (nach Confidence):")
for class_idx in range(num_classes):
    class_name = labels[class_idx]

    # Filter anwenden, falls gesetzt
    if FILTER_CLASSES is not None and class_name not in FILTER_CLASSES:
        print(f"   Überspringe '{class_name}' (nicht in FILTER_CLASSES)")
        continue

    # Finde alle Bilder, die zu dieser Klasse vorhergesagt wurden
    idx_cls = np.where(pred_classes == class_idx)[0]

    if len(idx_cls) == 0:
        print(f"   Keine Bilder für '{class_name}' vorhergesagt")
        continue

    # Sortiere nach Confidence (Logit-Wert) für diese Klasse
    sorted_idx = sorted(idx_cls, key=lambda i: pred_logits[i, class_idx], reverse=True)

    # Wähle Top-N aus
    num_selected = min(num_top, len(idx_cls))
    top_images_per_class[class_name] = [all_images[i] for i in sorted_idx[:num_selected]]

    # Zeige Confidence-Werte der Top-5
    top_confidences = [pred_logits[i, class_idx] for i in sorted_idx[:num_selected]]
    conf_str = ", ".join([f"{c:.2f}" for c in top_confidences])
    print(f"  ✓ {class_name}: {len(idx_cls)} Bilder, Top-{num_selected} ausgewählt")
    print(f"    Logits: [{conf_str}]")

if len(top_images_per_class) == 0:
    print("\n Keine Bilder zum Visualisieren gefunden!")
    print("Mögliche Gründe:")
    print("  - FILTER_CLASSES schließt alle Klassen aus")
    print("  - Keine Bilder wurden zu den gewünschten Klassen klassifiziert")
    exit()

# ============================================================
# 6) Grad-CAM vorbereiten
# ============================================================
activations = None
gradients = None

target_layer = model.swin.encoder.layers[-1].blocks[-1].layernorm_after

def forward_hook(module, inp, out):
    global activations
    activations = out.detach()

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0].detach()

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

print("\n Grad-CAM Hooks registriert")

# ============================================================
# 7) Top-5 Bilder pro Klasse visualisieren
# ============================================================
print("\n🎨 Erstelle Visualisierungen...")

num_classes_with_images = len(top_images_per_class)
max_cols = max(len(imgs) for imgs in top_images_per_class.values())

fig, axes = plt.subplots(num_classes_with_images, max_cols,
                         figsize=(4*max_cols, 4*num_classes_with_images))

# Sicherstellen, dass axes immer 2D ist
if num_classes_with_images == 1 and max_cols == 1:
    axes = np.array([[axes]])
elif num_classes_with_images == 1:
    axes = axes.reshape(1, -1)
elif max_cols == 1:
    axes = axes.reshape(-1, 1)

for row, class_name in enumerate(sorted(top_images_per_class.keys())):
    # Finde den Klassen-Index im Modell
    class_idx = [idx for idx, name in labels.items() if name == class_name][0]

    for col, img_path in enumerate(top_images_per_class[class_name]):
        print(f"  Verarbeite {class_name} - Bild {col+1}/{len(top_images_per_class[class_name])}")

        try:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(img, return_tensors="pt").to(device)

            # Forward
            outputs = model(**inputs)

            # Backward für Grad-CAM
            model.zero_grad()
            outputs.logits[0, class_idx].backward()

            # Grad-CAM berechnen
            grads = gradients.mean(dim=1)
            weights = grads[0]
            cam = (activations[0] * weights).sum(dim=-1)
            cam = F.relu(cam)

            # Normalisierung
            if cam.max() > 0:
                cam -= cam.min()
                cam /= cam.max()

            # Reshape und Resize
            grid_size = int(np.sqrt(cam.shape[0]))
            cam = cam.cpu().numpy().reshape(grid_size, grid_size)
            cam = cv2.resize(cam, (img.width, img.height))

            # Heatmap erstellen
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = np.clip(0.6 * np.array(img) + 0.4 * heatmap, 0, 255).astype(np.uint8)

            # Visualisierung
            ax = axes[row, col]
            ax.imshow(overlay)
            ax.axis('off')

            # Prediction Score anzeigen
            pred_score = torch.softmax(outputs.logits[0], dim=0)[class_idx].item()
            filename = os.path.basename(img_path)
            ax.set_title(f"{class_name.replace('_', ' ').title()}\n{filename[:25]}\nConfidence: {pred_score:.1%}",
                        fontsize=10, pad=5)

        except Exception as e:
            print(f"     Fehler bei Bild {img_path}: {e}")
            ax = axes[row, col]
            ax.axis('off')
            ax.text(0.5, 0.5, "Error", ha='center', va='center')

    # Leere Subplots ausblenden, falls weniger als max_cols Bilder
    for col in range(len(top_images_per_class[class_name]), max_cols):
        axes[row, col].axis('off')

plt.tight_layout()
output_path = '/content/grad_cam_top5_tumor_classes.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Visualisierung gespeichert: {output_path}")
plt.show()

print("\n Fertig!")
print(f" Visualisierte Klassen: {', '.join(sorted(top_images_per_class.keys()))}")



