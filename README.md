# KI-gestützte Klassifikation und Segmentierung von Gehirntumoren mittels MRT-Daten

##  Projektübersicht
Diese Maturaarbeit untersucht den Einsatz von künstlicher Intelligenz (KI) zur automatisierten **Klassifikation und Segmentierung von Gehirntumoren** anhand von Magnetresonanztomographie-(MRT)-Daten.  
Der Fokus liegt auf der Evaluation vortrainierter Machine-Learning-Modelle, dem Verständnis ihrer Funktionsweise sowie einer kritischen Analyse ihrer Limitationen im medizinischen Kontext.

Untersucht wurden:
- ein **Transformer-basiertes Klassifikationsmodell**
- ein **U-Net-basiertes Segmentierungsmodell**

---

##  Zielsetzung
Ziel der Arbeit war es:
- die Leistungsfähigkeit moderner KI-Modelle in der medizinischen Bildanalyse zu evaluieren,
- deren Eignung zur automatisierten Tumorerkennung zu untersuchen,
- Limitationen hinsichtlich Datenvielfalt, Robustheit, Interpretierbarkeit und klinischer Anwendbarkeit kritisch zu analysieren.

---

##  Verwendete Modelle

### Klassifikationsmodell
- **Name:** Brain Tumor Classification
- **Architektur:** Transformer-basiert
- **Quelle:**  
  Devarshi – Hugging Face  
  https://huggingface.co/Devarshi/Brain_Tumor_Classification

### Segmentierungsmodell
- **Name:** U-Net Brain Tumor Segmentation
- **Architektur:** U-Net
- **Quelle:**  
  bombshelll – Hugging Face  
  https://huggingface.co/bombshelll/unet-brain-tumor-segmentation/tree/main/variables

Beide Modelle wurden vortrainiert und für diese Arbeit weiter optimiert und evaluiert.

---

##  Datensätze

Aufgrund der Dateigrösse konnten die vollständigen Datensätze sowie das optimierte Modell nicht direkt auf GitHub hochgeladen werden.

###  Externe Download-Links

#### Optimiertes Modell
- https://drive.google.com/drive/folders/1S1A1v9tPvgonHW5u0rwP2oOdCCgTf2uj

#### Klassifikation
- **Trainingsdatenset:**  
  https://drive.google.com/drive/folders/1D5SbsDbrsGDJDZX-hAcwfemjKMpFotN5
- **Auswertungsdatenset (≈2000 Dateien):**  
  https://drive.google.com/drive/folders/1s3AkZARE1HfVVoi3oz_8SxP5564x39gC

#### Segmentierung
- **Originales Segmentierungsdatenset:**  
  https://drive.google.com/drive/folders/1X8-zfaEHYHMHzilfnmGee8nTysf__D8C
- **Konvertiertes Segmentierungsdatenset (≈3000 Dateien):**  
  https://drive.google.com/drive/folders/1RWacr3gXgwt4j3IxUWHLem18d4ujB3h9

---

##  Methodik

### Vorverarbeitung
- Normalisierung der MRT-Bilder
- Resizing zur Vereinheitlichung der Eingabedimensionen
- Konvertierung der Segmentierungsdaten aus `.mat`-Dateien
- TTA mit Drehen um 5-10 Grad, Kontrast verändern, Spiegelungen 

### Trainings- & Evaluationsumfang
| Modell | Trainingsdaten | Testdaten |
|------|---------------|-----------|
| Klassifikation | ca. 5000 Bilder (jpeg) | ca. 5000 Bilder (jpeg + png) |
| Segmentierung | - | ca. 3100 (.mat)Dateien |

### Evaluationsmetriken
- **Klassifikation:** Accuracy, Precison, F1-Score
- **Segmentierung:** Dice Score, Intersection over Union (IoU)
- Zusätzlich: qualitative visuelle Analyse von Segmentierungen, Fehlklassifikationen sowie die GRAD-CAM 

---

##  Ergebnisse

### Klassifikation
- **Testgenauigkeit:** **99.5 % / 95.6 %** mit dem kovertierten Datensatz
- Zuverlässige Unterscheidung von vier Tumorkategorien

### Segmentierung
- **Dice Score:** **86 %**
- Tumorregionen wurden mehrheitlich präzise segmentiert

Die Resultate zeigen das hohe Potenzial moderner KI-Modelle in der medizinischen Bildanalyse.

---

## Limitationen
Trotz der hohen Leistungswerte bestehen relevante Einschränkungen:
- Begrenzte Datenvielfalt
- Eingeschränkte Robustheit gegenüber unbekannten Daten
- Geringe Interpretierbarkeit der Modelle
- Eingeschränkte klinische Generalisierbarkeit

Ein klinischer Einsatz erfordert daher grössere Datensätze, transparente Modellarchitekturen und umfangreiche Validierungsstudien.

---

##  Repository-Struktur
```text
├── Brain_Tumor_Results/      # Tabellen und Auswertungen der Klassifikation 
├── KI/                       # Prompts und KI-generierter Code
├── Literatur/                # Fachliteratur, Links und Ressourcen
├── Resultate/                # Outputs der Entwicklungsumgebung
├── code/                     # Alle verwendeten Python-Skripte
├── .gitignore
├── Klassifizierug_Auswertung_Beispiel # Notebook zur Beispielklassifizierung 
├── LICENSE  
└── README.md


