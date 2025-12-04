# ============================================
# MODELL-INSPEKTION KLASSIFIZIERUNG 
# Finde heraus wie viele Layer das Modell hat
# ============================================

from transformers import AutoModelForImageClassification

print(" Inspektion der Modell-Architektur...\n")

model = AutoModelForImageClassification.from_pretrained(
    "Devarshi/Brain_Tumor_Classification"
)

print("="*80)
print(" MODELL-STRUKTUR")
print("="*80)

# Zeige Top-Level Struktur
print("\nTop-Level Module:")
for name, module in model.named_children():
    print(f"  - {name}: {type(module).__name__}")

# Finde die Basis-Architektur
if hasattr(model, 'vit'):
    base = model.vit
    print("\n ViT (Vision Transformer) erkannt")
elif hasattr(model, 'resnet'):
    base = model.resnet
    print("\n ResNet erkannt")
elif hasattr(model, 'convnext'):
    base = model.convnext
    print("\n ConvNeXt erkannt")
elif hasattr(model, 'swin'):
    base = model.swin
    print("\n Swin Transformer erkannt")
else:
    base = model
    print("\n Unbekannte Architektur")

# Zähle Encoder/Layer
print("\n" + "="*80)
print(" LAYER-ANALYSE")
print("="*80)

if hasattr(base, 'encoder'):
    if hasattr(base.encoder, 'layer'):
        num_layers = len(base.encoder.layer)
        print(f"\n Gefunden: {num_layers} Encoder-Layer")

        print("\nLayer-Details:")
        for i, layer in enumerate(base.encoder.layer):
            param_count = sum(p.numel() for p in layer.parameters())
            print(f"  Layer {i:2d}: {param_count:,} Parameter")

    elif hasattr(base.encoder, 'layers'):
        num_layers = len(base.encoder.layers)
        print(f"\n Gefunden: {num_layers} Encoder-Layer")
else:
    print("\n Keine 'encoder' Struktur gefunden")

# Zeige gesamte Parameter
print("\n" + "="*80)
print(" PARAMETER-STATISTIK")
print("="*80)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nGesamte Parameter: {total_params:,}")

# Zeige Embeddings
if hasattr(base, 'embeddings'):
    emb_params = sum(p.numel() for p in base.embeddings.parameters())
    print(f"Embedding Layer:   {emb_params:,} Parameter")

# Zeige Classifier
if hasattr(model, 'classifier'):
    clf_params = sum(p.numel() for p in model.classifier.parameters())
    print(f"Classifier:        {clf_params:,} Parameter")

# Empfehlung für Layer-Freezing
print("\n" + "="*80)
print(" EMPFEHLUNG FÜR FINE-TUNING")
print("="*80)

if hasattr(base, 'encoder') and hasattr(base.encoder, 'layer'):
    num_layers = len(base.encoder.layer)

    # Empfehlung: Friere 60-70% der Layer ein
    recommended_freeze = int(num_layers * 0.65)

    print(f"\nGesamte Layer:           {num_layers}")
    print(f"Empfohlene Freeze:       {recommended_freeze} Layer")
    print(f"Trainierbare Layer:      {num_layers - recommended_freeze}")

    print(f"\n Ändere in fine_tune_brain_tumor.py:")
    print(f"   'num_layers_to_freeze': {recommended_freeze}")

    # Alternative Strategien
    print("\nAlternative Strategien:")
    print(f"  Konservativ (70%):  {int(num_layers * 0.7)} Layer einfrieren")
    print(f"  Moderat (60%):      {int(num_layers * 0.6)} Layer einfrieren")
    print(f"  Aggressiv (50%):    {int(num_layers * 0.5)} Layer einfrieren")

print("\n" + "="*80)
print(" INSPEKTION ABGESCHLOSSEN")
print("="*80)
