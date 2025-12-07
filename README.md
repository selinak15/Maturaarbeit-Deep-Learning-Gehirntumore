# Maturaarbeit-Deep-Learning-Gehirntumore

Zusammenfassung: 
Die vorliegende Maturaarbeit untersucht die Anwendung von KI-Modellen zur Klassifikation und Segmentierung von Gehirntumoren mit MRT-Daten. Ziel der Arbeit war es, die Leistungsfähigkeit von vortrainierten Machine-Learning-Modellen zu evaluieren, deren Funktionsweise zu verstehen und deren Limitationen im medizinischen Kontext kritisch zu beurteilen. Im Fokus standen ein Transformer-basiertes Klassifikationsmodell sowie ein U-Net-basiertes Segmentierungsmodell.

Für die Experimente wurden zwei öffentlich verfügbare MRT-Datensätze verwendet, die zuvor mittels Normalisierung und Resizing vorverarbeitet wurden. Das Klassifikationsmodell wurde mit rund 5000 Bildern trainiert und mit 5000 Bildern getestet, während das U-Net-Modell auf 3100 mehrdimensionalen .mat-Dateien mit zugehörigen Tumormasken ausgewertet wurde. Die Modellleistung wurde mithilfe Metriken wie Accuracy, Dice Score und Intersection over Union (IoU) beurteilt. Zusätzlich wurden qualitative Analysen durchgeführt, indem beispielhafte Segmentierungen und Fehlklassifikationen visuell beurteilt wurden.

Das Klassifikationsmodell erreichte eine Testgenauigkeit von 99.5 % und konnte die vier Tumorkategorien zuverlässig unterscheiden. Das Segmentierungsmodell erzielte einen Dice Score von 86 %, womit die Tumorregionen mehrheitlich präzise erkannt wurden. Die Ergebnisse bestätigen, dass aktuelle KI-Modelle ein hohes Potenzial für die automatisierten Tumorerkennung besitzen. Gleichzeitig zeigten die Analysen, dass Limitationen bezüglich Datenvielfalt, Modellrobustheit, Interpretierbarkeit und klinischer Generalisierbarkeit bestehen.
Insgesamt verdeutlicht die Arbeit, dass KI-gestützte Bildanalyse bereits heute leistungsfähige Ergebnisse liefern kann, ihr klinischer Einsatz jedoch eine sorgfältige Validierung, grössere Datensätze und transparente Modellarchitekturen erfordert. 

Bei dieser Maturitätsarbeit wurden zwei vortrainierte ML-Modelle genutzt und optimiert für die Segmentierung und Klassifizierung von Gehirntumoren. 

genutzte Modelle: 
Devarshi/Brain_Tumor_Classification · Hugging Face. (n.d.). Huggingface.co. https://huggingface.co/Devarshi/Brain_Tumor_Classification

bombshelll/unet-brain-tumor-segmentation at main. (2025, April 21). Huggingface.co. https://huggingface.co/bombshelll/unet-brain-tumor-segmentation/tree/main/variables

Leider konnte das optimierte Modell sowie die vollständigen Datensets nicht auf Git Hub geladen werden augfrund der Grösse der Datei: 

Hier finden Sie einen Link zum optimierten Modell: 
https://drive.google.com/drive/folders/1S1A1v9tPvgonHW5u0rwP2oOdCCgTf2uj?usp=share_link


Die Links zum Datenset: 

Auswertungsdatenset Klassifizierung: ca. 2000 Dateien
https://drive.google.com/drive/folders/1s3AkZARE1HfVVoi3oz_8SxP5564x39gC?usp=share_link

konvertiertes Segmentierungsdatenset: (ca. 3000 Dateien)
https://drive.google.com/drive/folders/1RWacr3gXgwt4j3IxUWHLem18d4ujB3h9?usp=share_link

Trainingsdatenset für die Klassifizierung: 
https://drive.google.com/drive/folders/1D5SbsDbrsGDJDZX-hAcwfemjKMpFotN5?usp=share_link

Segmentierungsdatenset: 
https://drive.google.com/drive/folders/1X8-zfaEHYHMHzilfnmGee8nTysf__D8C?usp=share_link

Falls Sie die Links beziehungsweise die Datensets benutzen, bitte zitieren Sie: 
Segmentierung: Cheng, J., Huang, W., Cao, S., Yang, R., Yang, W., Yun, Z., Wang, Z., & Feng, Q. (2015). Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition. PLOS ONE, 10(10), e0140381. 

Klassifizierung: Masoud Nickparvar. (2021). Brain Tumor MRI Dataset. Doi.org. https://doi.org/10.34740/KAGGLE/DSV/2645886

Im Literaturordner sind nützliche Ressourcen beschrieben und Links angegeben. Im KI-Ordner sind alle Prompts zu finden, sowie alle KI generierten Codes. Im Codeordner sind alle verendeten Codes zu finden. Im Resultateordner sind die Outputs der Entwicklungsumgebung zu finden. Im Brain_Tumor_Results sind die Tabellen der Auswertungsversuche der Klassifikation. 
