"""
============================================
FINE-TUNE U-NET SEGMENTATION MODEL V2
Based on actual model inspection results
============================================

Model: bombshelll/unet-brain-tumor-segmentation
- 31M parameters
- Has Batch Normalization
- Input: keras_tensor_68 (256×256×3)
- Output: output_0 (256×256×1)

Expected improvement: 86.57% → 92-95% Dice
Training time: 1-2 hours on T4 GPU
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import json
from datetime import datetime
from collections import defaultdict
import random
from huggingface_hub import snapshot_download

print("="*70)
print("🚀 BRAIN TUMOR SEGMENTATION FINE-TUNING V2")
print("="*70)

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    # Paths
    'data_dir': '/content/drive/My Drive/Tumor/2',
    'output_dir': '/content/drive/My Drive/Brain_Tumor_Segmentation_FineTuned',

    # Model
    'base_model_repo': 'bombshelll/unet-brain-tumor-segmentation',

    # Training hyperparameters
    'img_size': 256,
    'batch_size': 8,
    'learning_rate': 5e-5,  # Lower than before (31M params is large)
    'num_epochs': 20,
    'samples_per_class': 700,
    'validation_split': 0.15,
    'test_split': 0.15,

    # Data augmentation (medical-safe only!)
    'use_augmentation': True,
    'horizontal_flip': True,
    'brightness_range': 0.1,
    'zoom_range': 0.05,

    # Transfer learning strategy
    'freeze_strategy': 'encoder_first_half',  # Options: 'none', 'encoder_first_half', 'encoder_full'
    'freeze_batch_norm': True,  # Keep BN statistics from pre-training

    # Early stopping
    'patience': 7,
    'min_delta': 0.001,
}

print(f"\n📋 CONFIGURATION:")
print(f"   Data Directory: {CONFIG['data_dir']}")
print(f"   Output Directory: {CONFIG['output_dir']}")
print(f"   Batch Size: {CONFIG['batch_size']}")
print(f"   Learning Rate: {CONFIG['learning_rate']}")
print(f"   Epochs: {CONFIG['num_epochs']}")
print(f"   Samples per Class: {CONFIG['samples_per_class']}")
print(f"   Freeze Strategy: {CONFIG['freeze_strategy']}")

# ============================================
# 1. DATA LOADING (Same as before)
# ============================================

def load_mat_file(file_path):
    """Load .mat file (handles both v7.3 and older formats)"""
    try:
        with h5py.File(file_path, 'r') as f:
            if 'cjdata' in f:
                image = np.array(f['cjdata']['image']).T
                mask = np.array(f['cjdata']['tumorMask']).T
                label = int(np.array(f['cjdata']['label']).item())
                return {
                    'image': image.astype(np.float32),
                    'tumorMask': mask.astype(np.float32),
                    'label': label
                }
    except:
        pass

    try:
        from scipy.io import loadmat
        mat = loadmat(file_path)
        if 'cjdata' in mat:
            cjdata = mat['cjdata'][0, 0]
            return {
                'image': cjdata['image'].astype(np.float32),
                'tumorMask': cjdata['tumorMask'].astype(np.float32),
                'label': int(cjdata['label'])
            }
    except:
        pass

    return None

def load_dataset(data_dir, samples_per_class=700, verbose=True):
    """Load and balance dataset"""
    all_files = glob.glob(os.path.join(data_dir, '**/*.mat'), recursive=True)

    if verbose:
        print(f"\n📁 Found {len(all_files)} .mat files")

    class_files = {1: [], 2: [], 3: []}
    tumor_types = {1: "Meningiom", 2: "Gliom", 3: "Hypophysentumor"}

    print("🔍 Scanning dataset...")
    for file in tqdm(all_files, desc="Scanning labels"):
        try:
            data = load_mat_file(file)
            if data:
                label = data['label']
                if label in [1, 2, 3]:
                    class_files[label].append(file)
        except:
            continue

    if verbose:
        print(f"\n📊 Class distribution:")
        for cls in [1, 2, 3]:
            print(f"   Class {cls} ({tumor_types[cls]}): {len(class_files[cls])} files")

    balanced_files = []
    for cls in [1, 2, 3]:
        if class_files[cls]:
            n_samples = min(samples_per_class, len(class_files[cls]))
            random.shuffle(class_files[cls])
            balanced_files.extend(class_files[cls][:n_samples])
            if verbose:
                print(f"   → Sampling {n_samples} from class {cls}")

    random.shuffle(balanced_files)

    if verbose:
        print(f"\n✅ Balanced dataset: {len(balanced_files)} total samples")

    X, y, labels = [], [], []

    print("\n📥 Loading images...")
    for file in tqdm(balanced_files, desc="Loading data"):
        try:
            data = load_mat_file(file)
            if not data:
                continue

            image = data['image']
            mask = data['tumorMask']
            label = data['label']

            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            mask = (mask > 0).astype(np.float32)

            X.append(image)
            y.append(mask)
            labels.append(label)
        except:
            continue

    X = np.array(X)
    y = np.array(y)
    labels = np.array(labels)

    if verbose:
        print(f"\n✅ Loaded dataset:")
        print(f"   Images shape: {X.shape}")
        print(f"   Masks shape: {y.shape}")
        print(f"   Labels shape: {labels.shape}")

    return X, y, labels

# ============================================
# 2. DATA AUGMENTATION (MEDICAL-SAFE)
# ============================================

def augment_image_and_mask(image, mask, config):
    """Medical-safe augmentation only"""
    if config['horizontal_flip'] and np.random.random() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)

    if np.random.random() > 0.5:
        factor = 1.0 + np.random.uniform(-config['brightness_range'], config['brightness_range'])
        image = np.clip(image * factor, 0, 1)

    if config['zoom_range'] > 0 and np.random.random() > 0.5:
        zoom_factor = 1.0 + np.random.uniform(-config['zoom_range'], config['zoom_range'])
        h, w = image.shape

        crop_h = int(h / zoom_factor)
        crop_w = int(w / zoom_factor)

        start_h = np.random.randint(0, h - crop_h + 1) if h > crop_h else 0
        start_w = np.random.randint(0, w - crop_w + 1) if w > crop_w else 0

        image_crop = image[start_h:start_h+crop_h, start_w:start_w+crop_w]
        mask_crop = mask[start_h:start_h+crop_h, start_w:start_w+crop_w]

        image = cv2.resize(image_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask_crop, (w, h), interpolation=cv2.INTER_NEAREST)

    return image, mask

# ============================================
# 3. CUSTOM DATA GENERATOR
# ============================================

class SegmentationDataGenerator(keras.utils.Sequence):
    """Custom data generator with medical-safe augmentation"""

    def __init__(self, X, y, batch_size, img_size, config, augment=False, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.img_size = img_size
        self.config = config
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_X = []
        batch_y = []

        for i in indices:
            img = self.X[i].copy()
            mask = self.y[i].copy()

            if self.augment:
                img, mask = augment_image_and_mask(img, mask, self.config)

            if img.shape != (self.img_size, self.img_size):
                img_resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            else:
                img_resized = img
                mask_resized = mask

            img_rgb = np.stack([img_resized, img_resized, img_resized], axis=-1)

            batch_X.append(img_rgb)
            batch_y.append(np.expand_dims(mask_resized, -1))

        return np.array(batch_X, dtype=np.float32), np.array(batch_y, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ============================================
# 4. CUSTOM LOSSES & METRICS
# ============================================

def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice loss (1 - Dice coefficient)"""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return 1 - dice

def combined_loss(y_true, y_pred):
    """Combined Dice + Binary Cross-Entropy"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

def dice_coefficient(y_true, y_pred, threshold=0.5, smooth=1e-6):
    """Dice coefficient metric"""
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_binary, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return dice

def iou_score(y_true, y_pred, threshold=0.5, smooth=1e-6):
    """IoU metric"""
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_binary, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# ============================================
# 5. LOAD AND CONFIGURE PRE-TRAINED MODEL
# ============================================

def load_and_configure_model(config):
    """
    Load pre-trained model from Hugging Face and configure for fine-tuning

    Based on inspection:
    - conv2d_19 to conv2d_28: Encoder (first half to freeze)
    - conv2d_29 to conv2d_37: Decoder (always train)
    - Batch normalization layers present
    """
    print("\n" + "="*70)
    print("LOADING PRE-TRAINED MODEL")
    print("="*70)

    # Download model
    print("\n1️⃣  Downloading from Hugging Face...")
    model_path = snapshot_download(repo_id=config['base_model_repo'])
    print(f"   ✅ Downloaded to: {model_path}")

    # Load SavedModel
    print("\n2️⃣  Loading SavedModel...")
    saved_model = tf.saved_model.load(model_path)
    print("   ✅ Loaded successfully")

    # Get the inference function
    infer = saved_model.signatures['serving_default']

    # Create a Keras Model wrapper for fine-tuning
    print("\n3️⃣  Converting to Keras model...")

    # Define input
    inputs = keras.Input(shape=(256, 256, 3), name='input_image')

    # Apply the pre-trained model
    outputs = infer(keras_tensor_68=inputs)['output_0']

    # Create Keras model
    model = keras.Model(inputs=inputs, outputs=outputs, name='unet_finetuned')

    print("   ✅ Keras model created")
    print(f"   Total params: {model.count_params():,}")

    # Apply freezing strategy
    print("\n4️⃣  Applying freezing strategy...")
    freeze_strategy = config['freeze_strategy']

    if freeze_strategy == 'encoder_first_half':
        print("   🔒 Freezing first half of encoder (conv2d_19 to conv2d_24)")
        # This will freeze early feature extractors
        freeze_layer_range = range(19, 25)  # conv2d_19 to conv2d_24

    elif freeze_strategy == 'encoder_full':
        print("   🔒 Freezing entire encoder (conv2d_19 to conv2d_28)")
        freeze_layer_range = range(19, 29)  # conv2d_19 to conv2d_28

    elif freeze_strategy == 'none':
        print("   🔓 No freezing - training all layers")
        freeze_layer_range = []

    frozen_count = 0
    for layer in model.layers:
        layer_name = layer.name

        # Check if this is a conv2d layer in the freeze range
        if 'conv2d_' in layer_name:
            try:
                layer_num = int(layer_name.split('_')[1])
                if layer_num in freeze_layer_range:
                    layer.trainable = False
                    frozen_count += 1
                    print(f"      ❄️  Frozen: {layer_name}")
            except:
                pass

        # Also freeze corresponding batch normalization layers
        if config['freeze_batch_norm'] and 'batch_normalization_' in layer_name:
            try:
                bn_num = int(layer_name.split('_')[2])
                # BN layers are numbered similarly to conv layers
                if (bn_num + 1) in freeze_layer_range:  # Offset by 1
                    layer.trainable = False
                    frozen_count += 1
            except:
                pass

    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])

    print(f"\n   ✅ Freezing complete:")
    print(f"      Frozen layers: {frozen_count}")
    print(f"      Trainable params: {trainable_params:,}")
    print(f"      Non-trainable params: {non_trainable_params:,}")

    return model

# ============================================
# 6. TRAINING FUNCTION
# ============================================

def train_segmentation_model(config):
    """Main training function"""

    print("\n" + "="*70)
    print("STEP 1: LOADING DATASET")
    print("="*70)

    X, y, labels = load_dataset(
        config['data_dir'],
        samples_per_class=config['samples_per_class'],
        verbose=True
    )

    # Split dataset
    print("\n" + "="*70)
    print("STEP 2: SPLITTING DATASET")
    print("="*70)

    X_train, X_temp, y_train, y_temp, labels_train, labels_temp = train_test_split(
        X, y, labels,
        test_size=(config['validation_split'] + config['test_split']),
        random_state=42,
        stratify=labels
    )

    val_ratio = config['validation_split'] / (config['validation_split'] + config['test_split'])
    X_val, X_test, y_val, y_test, labels_val, labels_test = train_test_split(
        X_temp, y_temp, labels_temp,
        test_size=(1 - val_ratio),
        random_state=42,
        stratify=labels_temp
    )

    print(f"\n📊 Dataset split:")
    print(f"   Training:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    # Create data generators
    print("\n" + "="*70)
    print("STEP 3: CREATING DATA GENERATORS")
    print("="*70)

    train_gen = SegmentationDataGenerator(
        X_train, y_train, config['batch_size'], config['img_size'],
        config, augment=config['use_augmentation'], shuffle=True
    )

    val_gen = SegmentationDataGenerator(
        X_val, y_val, config['batch_size'], config['img_size'],
        config, augment=False, shuffle=False
    )

    test_gen = SegmentationDataGenerator(
        X_test, y_test, config['batch_size'], config['img_size'],
        config, augment=False, shuffle=False
    )

    print(f"   Training batches: {len(train_gen)}")
    print(f"   Validation batches: {len(val_gen)}")
    print(f"   Test batches: {len(test_gen)}")
    print(f"   Augmentation: {'✅ ENABLED' if config['use_augmentation'] else '❌ DISABLED'}")

    # Load pre-trained model
    model = load_and_configure_model(config)

    # Compile model
    print("\n" + "="*70)
    print("STEP 4: COMPILING MODEL")
    print("="*70)

    optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[dice_coefficient, iou_score, 'binary_accuracy']
    )

    print("✅ Model compiled")
    print(f"   Optimizer: Adam (lr={config['learning_rate']})")
    print(f"   Loss: Combined (Dice + BCE)")

    # Callbacks
    print("\n" + "="*70)
    print("STEP 5: SETTING UP CALLBACKS")
    print("="*70)

    os.makedirs(config['output_dir'], exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient',
            patience=config['patience'],
            min_delta=config['min_delta'],
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_dice_coefficient',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config['output_dir'], 'best_model.h5'),
            monitor='val_dice_coefficient',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(config['output_dir'], 'training_log.csv')
        )
    ]

    print(f"✅ Callbacks configured")

    # Train
    print("\n" + "="*70)
    print("STEP 6: TRAINING MODEL")
    print("="*70)
    print(f"   Epochs: {config['num_epochs']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config['num_epochs'],
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    print("\n" + "="*70)
    print("STEP 7: SAVING MODEL")
    print("="*70)

    final_model_path = os.path.join(config['output_dir'], 'final_model')
    model.save(final_model_path)
    print(f"✅ Model saved to: {final_model_path}")

    # Plot training history
    print("\n" + "="*70)
    print("STEP 8: PLOTTING TRAINING HISTORY")
    print("="*70)

    plot_training_history(history, config['output_dir'])

    # Final evaluation
    print("\n" + "="*70)
    print("STEP 9: FINAL EVALUATION ON TEST SET")
    print("="*70)

    test_results = model.evaluate(test_gen, verbose=0)
    test_metrics = dict(zip(model.metrics_names, test_results))

    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Final Test Set Metrics:")
    print(f"   Dice Coefficient: {test_metrics['dice_coefficient']:.4f}")
    print(f"   IoU Score: {test_metrics['iou_score']:.4f}")
    print(f"   Binary Accuracy: {test_metrics['binary_accuracy']:.4f}")
    print(f"   Loss: {test_metrics['loss']:.4f}")
    print(f"{'='*70}")

    save_metrics_summary(history, test_metrics, config)

    return model, history, test_metrics

# ============================================
# 7. VISUALIZATION & SAVING
# ============================================

def plot_training_history(history, output_dir):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].plot(history.history['dice_coefficient'], label='Train Dice', linewidth=2)
    axes[0, 0].plot(history.history['val_dice_coefficient'], label='Val Dice', linewidth=2)
    axes[0, 0].set_title('Dice Coefficient', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Dice')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 1].set_title('Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history.history['iou_score'], label='Train IoU', linewidth=2)
    axes[1, 0].plot(history.history['val_iou_score'], label='Val IoU', linewidth=2)
    axes[1, 0].set_title('IoU Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history.history['binary_accuracy'], label='Train Acc', linewidth=2)
    axes[1, 1].plot(history.history['val_binary_accuracy'], label='Val Acc', linewidth=2)
    axes[1, 1].set_title('Binary Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Fine-Tuning Progress', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved: {plot_path}")
    plt.close()

def save_metrics_summary(history, test_metrics, config):
    """Save metrics to JSON"""
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'epochs_trained': len(history.history['loss']),
            'freeze_strategy': config['freeze_strategy'],
            'augmentation': config['use_augmentation']
        },
        'test_metrics': {
            'dice_coefficient': float(test_metrics['dice_coefficient']),
            'iou_score': float(test_metrics['iou_score']),
            'binary_accuracy': float(test_metrics['binary_accuracy']),
            'loss': float(test_metrics['loss'])
        },
        'best_validation': {
            'dice_coefficient': float(max(history.history['val_dice_coefficient'])),
            'epoch': int(np.argmax(history.history['val_dice_coefficient']) + 1)
        }
    }

    path = os.path.join(config['output_dir'], 'training_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved: {path}")

# ============================================
# 8. MAIN EXECUTION
# ============================================

if __name__ == '__main__':
    # Check GPU
    print("\n🔍 Checking GPU...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️  No GPU! Training will be very slow.")

    # Mount Drive
    print("\n📁 Mounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Drive mounted")
    except:
        print("⚠️  Not in Colab or already mounted")

    # Train
    print("\n" + "="*70)
    print("🚀 STARTING FINE-TUNING")
    print("="*70)

    model, history, test_metrics = train_segmentation_model(CONFIG)

    print("\n" + "="*70)
    print("✅ FINE-TUNING COMPLETE!")
    print("="*70)
    print(f"\n📁 Results: {CONFIG['output_dir']}")
    print(f"\n💡 Baseline: 86.57% Dice")
    print(f"💡 Your result: {test_metrics['dice_coefficient']:.2%} Dice")
    print(f"💡 Improvement: {(test_metrics['dice_coefficient'] - 0.8657)*100:+.2f}%")
