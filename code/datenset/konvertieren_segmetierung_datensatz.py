# ============================================
# MAT FILE TO IMAGE CONVERTER
# Konvertiert .mat Dateien zu Bildern mit Label im Dateinamen
# ============================================

from google.colab import drive
drive.mount('/content/drive')

import h5py
import numpy as np
from PIL import Image
import os
import glob
from pathlib import Path

print("="*80)
print(" MAT FILE TO IMAGE CONVERTER")
print("="*80)

# ============================================
# 1. KONFIGURATION
# ============================================

CONFIG = {
    'input_dir': '/content/drive/My Drive/Testing_Segmentation/forth',
    'output_dir': '/content/drive/My Drive/Testing_Segmentation/images',

    # Label Mapping (basierend auf deinem code_1.py)
    'label_mapping': {
        1: 'meningioma',  # Anpassen falls nötig
        2: 'glioma',       # Anpassen falls nötig
        3: 'pituitary',    # Anpassen falls nötig
        # Füge weitere Labels hinzu falls vorhanden
    }
}

print(f"\n Input:  {CONFIG['input_dir']}")
print(f" Output: {CONFIG['output_dir']}")

# Erstelle Output-Verzeichnis
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ============================================
# 2. FINDE ALLE .MAT DATEIEN
# ============================================

print("\n Suche .mat Dateien...")
mat_files = glob.glob(os.path.join(CONFIG['input_dir'], '*.mat'))
mat_files = sorted(mat_files)

print(f" Gefunden: {len(mat_files)} .mat Dateien")

if len(mat_files) == 0:
    print(" Keine .mat Dateien gefunden!")
    print(f"   Überprüfe Pfad: {CONFIG['input_dir']}")
    import sys
    sys.exit()

# ============================================
# 3. KONVERTIERUNG
# ============================================

print("\n🚀 Starte Konvertierung...")

successful = 0
failed = 0
skipped = 0
label_counts = {label_name: 0 for label_name in CONFIG['label_mapping'].values()}
label_counts['unknown'] = 0

for i, mat_path in enumerate(mat_files, 1):
    try:
        if i % 10 == 0 or i == len(mat_files):
            print(f"⏳ Verarbeitung: {i}/{len(mat_files)} ({i/len(mat_files)*100:.1f}%)")

        filename = os.path.basename(mat_path)
        file_id = os.path.splitext(filename)[0]  # z.B. "100" aus "100.mat"

        # Öffne .mat Datei
        with h5py.File(mat_path, 'r') as f:
            # Extrahiere Daten
            cj = f['cjdata']

            # 1. Patienten-ID
            try:
                pid = np.array(cj["PID"])
                pid_str = "".join(chr(c[0]) for c in pid)
            except:
                pid_str = file_id

            # 2. Bild
            image = np.array(cj['image'])

            # 3. Label (Tumor-Typ)
            label = np.array(cj['label'])

            # Konvertiere Label zu Integer
            if label.size == 1:
                label_int = int(label.flatten()[0])
            else:
                label_int = int(label[0])

            # Mappe Label zu Tumor-Typ
            if label_int in CONFIG['label_mapping']:
                tumor_type = CONFIG['label_mapping'][label_int]
                label_counts[tumor_type] += 1
            else:
                tumor_type = 'unknown'
                label_counts['unknown'] += 1
                print(f"    Unbekanntes Label {label_int} in {filename}")

            # Normalisiere Bild auf 0-255
            if image.dtype != np.uint8:
                # Normalisiere auf [0, 1] dann auf [0, 255]
                image = image.astype(float)
                image = (image - image.min()) / (image.max() - image.min() + 1e-10)
                image = (image * 255).astype(np.uint8)

            # Erstelle Dateinamen: {tumor_type}_{patient_id}_{file_id}.png
            output_filename = f"{tumor_type}_{pid_str}_{file_id}.png"
            output_path = os.path.join(CONFIG['output_dir'], output_filename)

            # Konvertiere zu PIL Image und speichere
            pil_image = Image.fromarray(image)

            # Konvertiere zu RGB falls Grayscale
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            pil_image.save(output_path)
            successful += 1

    except Exception as e:
        print(f"    Fehler bei {filename}: {e}")
        failed += 1
        continue

# ============================================
# 4. ZUSAMMENFASSUNG
# ============================================

print("\n" + "="*80)
print("📊 KONVERTIERUNGS-ZUSAMMENFASSUNG")
print("="*80)

print(f"\n✅ Erfolgreich:  {successful}")
print(f" Fehlgeschlagen: {failed}")
print(f"Total:          {len(mat_files)}")

print("\n Label-Verteilung:")
print("─" * 40)
for label_name, count in sorted(label_counts.items()):
    if count > 0:
        print(f"  {label_name:15}: {count:4} Bilder")

print("\n Gespeicherte Bilder:")
print(f"  {CONFIG['output_dir']}")

# Überprüfe ob Bilder tatsächlich gespeichert wurden
saved_images = glob.glob(os.path.join(CONFIG['output_dir'], '*.png'))
print(f"\n✅ Verifizierung: {len(saved_images)} PNG-Dateien im Output-Ordner")

# ============================================
# 5. BEISPIEL-AUSGABE
# ============================================

if len(saved_images) > 0:
    print("\n📝 Beispiel-Dateinamen:")
    for example in saved_images[:5]:
        print(f"  {os.path.basename(example)}")

print("\n" + "="*80)
print("✅ KONVERTIERUNG ABGESCHLOSSEN!")
print("="*80)
