# ============================================
# SEGMENTATION IMPROVEMENTS
# Add to your segmentation_code.py
# ============================================

import tensorflow as tf
import numpy as np

# ============================================
# IMPROVEMENT 1: AUTO-DETECT INPUT NAME
# ============================================

class GrayscaleToRGBModelWrapper:
    """Improved wrapper with auto-detection"""
    def __init__(self, model):
        self.model = model
        self.serving_fn = model.signatures.get('serving_default')

        # Auto-detect input name (no hardcoding!)
        input_signature = self.serving_fn.structured_input_signature[1]
        self.input_name = list(input_signature.keys())[0]
        print(f"✅ Auto-detected input name: {self.input_name}")

    def predict(self, x):
        if x.shape[-1] == 1:
            x_rgb = np.repeat(x, 3, axis=-1)
        else:
            x_rgb = x

        x_tensor = tf.constant(x_rgb, dtype=tf.float32)
        result = self.serving_fn(**{self.input_name: x_tensor})
        output_key = list(result.keys())[0]
        return result[output_key].numpy()

# ============================================
# IMPROVEMENT 2: TEST-TIME AUGMENTATION FOR SEGMENTATION
# ============================================

class SegmentationTTA:
    """Test-Time Augmentation für bessere Segmentierung"""

    @staticmethod
    def apply_augmentations(image):
        """
        Erstellt augmentierte Versionen des Bildes
        Returns: List of (augmented_image, reverse_transform_fn)
        """
        augs = []

        # 1. Original
        augs.append((
            image,
            lambda x: x  # Keine Transformation
        ))

        # 2. Horizontal Flip
        augs.append((
            np.fliplr(image),
            lambda x: np.fliplr(x)  # Reverse: flip zurück
        ))

        # 3. Vertical Flip
        augs.append((
            np.flipud(image),
            lambda x: np.flipud(x)
        ))

        # 4. Rotation 90°
        augs.append((
            np.rot90(image, k=1),
            lambda x: np.rot90(x, k=-1)  # Reverse: -90°
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
            confidence: Pixel-wise confidence (std deviation)
        """
        # Ensure correct shape
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        h, w = image.shape[:2]

        # Get augmentations
        augmentations = SegmentationTTA.apply_augmentations(image)

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
        avg_prediction = np.mean(predictions_stack, axis=0)

        # Confidence: inverse of std (lower std = higher confidence)
        confidence = 1.0 - np.std(predictions_stack, axis=0)

        # Apply threshold
        final_mask = (avg_prediction > threshold).astype(np.float32)

        return final_mask, confidence, avg_prediction

# ============================================
# IMPROVEMENT 3: UPDATED EVALUATION FUNCTION WITH TTA
# ============================================

def evaluate_with_tta(wrapped_model, balanced_files, class_distribution, use_tta=True):
    """
    Evaluation mit optionalem TTA

    Args:
        use_tta: If True, uses 6x TTA for better results
    """
    from tqdm import tqdm

    sm = SegmentationMetrics()  # Assumes you have this class
    all_metrics = {
        'dice': [], 'iou': [],
        'precision': [], 'recall': [],
        'hausdorff': [],
        'confidence': [],  # NEW: Track confidence
        'per_class': {1: [], 2: [], 3: []}
    }

    tumor_types = {1: "Meningiom", 2: "Gliom", 3: "Hypophysentumor"}

    print(f"\n🚀 STARTE EVALUATION {'MIT TTA (6x)' if use_tta else 'OHNE TTA'}")
    print(f"   Total Dateien: {len(balanced_files)}")
    print()

    successful = 0
    failed = 0
    class_counts = {1: 0, 2: 0, 3: 0}

    with tqdm(total=len(balanced_files), desc="Evaluation") as pbar:
        for i, mat_file in enumerate(balanced_files):
            try:
                # Load data (use your existing load_mat_file function)
                data = load_mat_file(mat_file)

                if data is None:
                    failed += 1
                    pbar.update(1)
                    continue

                image = data['image'].astype(np.float32)
                gt_mask = data['tumorMask'].astype(np.float32)
                label = data['label']
                class_counts[label] += 1

                # Normalize
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)

                # PREDICTION WITH OR WITHOUT TTA
                if use_tta:
                    pred_mask, confidence, soft_mask = SegmentationTTA.predict_with_tta(
                        wrapped_model, image, threshold=0.5
                    )
                    avg_confidence = np.mean(confidence)
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

                # Calculate metrics
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

                # Memory cleanup
                if successful % 25 == 0:
                    tf.keras.backend.clear_session()
                    import gc
                    gc.collect()

            except Exception as e:
                print(f"\nFehler bei {mat_file}: {str(e)[:100]}")
                failed += 1
                pbar.update(1)
                continue

    print(f"\n✅ Evaluation abgeschlossen!")
    print(f"   Erfolgreich: {successful} Bilder")
    print(f"   Fehlgeschlagen: {failed} Bilder")

    if use_tta and all_metrics['confidence']:
        print(f"\n📊 TTA Confidence: {np.mean(all_metrics['confidence']):.4f} ± {np.std(all_metrics['confidence']):.4f}")

    return all_metrics, successful, class_counts

# ============================================
# IMPROVEMENT 4: ADAPTIVE THRESHOLDING
# ============================================

def find_optimal_threshold(soft_masks, gt_masks):
    """
    Findet optimalen Threshold für Dice Score

    Args:
        soft_masks: List of soft prediction masks (probability maps)
        gt_masks: List of ground truth masks

    Returns:
        optimal_threshold: Best threshold value
    """
    thresholds = np.linspace(0.3, 0.7, 41)  # Test 0.30 to 0.70
    best_dice = 0
    best_threshold = 0.5

    for thresh in thresholds:
        dice_scores = []
        for soft_mask, gt_mask in zip(soft_masks, gt_masks):
            pred_mask = (soft_mask > thresh).astype(np.float32)
            dice = SegmentationMetrics.dice_coefficient(pred_mask, gt_mask)
            dice_scores.append(dice)

        avg_dice = np.mean(dice_scores)
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_threshold = thresh

    print(f"🎯 Optimal Threshold: {best_threshold:.3f} (Dice: {best_dice:.4f})")
    return best_threshold

# ============================================
# USAGE EXAMPLE
# ============================================

"""
# Replace your evaluation call with:

# 1. WITHOUT TTA (current approach)
metrics_no_tta, n_success, class_counts = evaluate_with_tta(
    wrapped_model, balanced_files, class_distribution, use_tta=False
)

# 2. WITH TTA (improved - expected +2-5% Dice improvement)
metrics_with_tta, n_success, class_counts = evaluate_with_tta(
    wrapped_model, balanced_files, class_distribution, use_tta=True
)

# Expected results:
# - Without TTA: Dice ~0.75-0.85
# - With TTA: Dice ~0.78-0.88 (+3-5%)
# - Slower: 6x computation time
"""
