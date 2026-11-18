import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as FF
from PIL import Image
from tqdm import tqdm

from utils.cityscapes import Cityscapes
from utils.onn_prop_partfc import ONN_2class


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Evaluation defaults. Edit these values directly when running from the IDE.
EXP_NAME = 'segmentation_heatmap'
EVAL_DATASET = False
DATA_ROOT = "data/cityscapes"
SPLIT = 'train'
MAX_IMAGES = 500
OUTPUT_JSON = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SORT_DATASET = True
USE_SMOOTH = True
SMOOTH_CLASSES = ('car', 'human', 'vegetation')
SMOOTH_KERNEL_SIZE = 30
SMOOTH_REMOVE_ISOLATED_SIZE = 16
SAVE_VIS_NUM = 10
SAVE_VIS_DIR = None
CROP_TO_768 = True
CROP_SIZE = 768
# Used only when EVAL_DATASET is False.
IMAGE_PATH = 'data/cityscapes/leftImg8bit/train/aachen/aachen_000084_000019_leftImg8bit.png'

MODEL_CLASSES = ['sky', 'building', 'vegetation', 'road', 'sidewalk', 'traffic', 'car', 'human']
CLASS_MIN_PATCH_SIZES = {
    'sky': 32,
    'building': 32,
    'vegetation': 32,
    'road': 32,
    'sidewalk': 32,
    'traffic': 32,
    'car': 32,
    'human': 32,
}
CLASS_COLORS = {
    'car': (142, 0, 0),
    'human': (60, 20, 220),
    'traffic': (220, 220, 0),
    'sky': (70, 130, 180),
    'road': (128, 64, 128),
    'building': (70, 70, 70),
    'vegetation': (107, 142, 35),
    'sidewalk': (244, 35, 232),
}
CITYSCAPES_CLASS_IDS = {
    'road': (0,),
    'sidewalk': (1,),
    'building': (2, 3),
    'traffic': (6, 7),
    'vegetation': (8,),
    'sky': (10,),
    'human': (11, 12),
    'car': (13,),
}
HEATMAP_THRESHOLDS = {
    'sky': 0.50,
    'building': 0.50,
    'vegetation': 0.50,
    'road': 0.50,
    'sidewalk': 0.50,
    'traffic': 0.50,
    'car': 0.50,
    'human': 0.3,
}
CLASS_SCORE_WEIGHTS = {
    'sky': 1.0,
    'building': 1.0,
    'vegetation': 1.0,
    'road': 1.0,
    'sidewalk': 1.0,
    'traffic': 1.0,
    'car': 1.0,
    'human': 1.5,
}
PATCH_STRIDES = {
    128: 64,
    64: 32,
    32: 16,
}
PATCH_WEIGHTS = {
    128: 1.0,
    64: 1.0,
    32: 1.0,
}
BATCH_SIZE = 32
NORMALIZE_HEATMAP = True
MERGE_MODES = ('order', 'heatmap_score', 'heatmap_argmax')
SAVE_HEATMAP_COLORBAR = True
SAVE_HEATMAP_NUMPY = True
HEATMAP_VIS_MIN = 0.0
HEATMAP_VIS_MAX = 1.0
SAVE_HEATMAP_NUM = 0
SAVE_HEATMAP_DIR = None


def get_result_dir(exp_name):
    return os.path.join(SCRIPT_DIR, 'result_seg_heatmap', exp_name)


def sort_cityscapes_dataset(dataset):
    pairs = sorted(zip(dataset.images, dataset.targets), key=lambda item: item[0])
    dataset.images = [image for image, _ in pairs]
    dataset.targets = [target for _, target in pairs]
    return dataset


def center_crop_pil(image, crop_size=CROP_SIZE):
    width, height = image.size
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size))


def center_crop_array(array, crop_size=CROP_SIZE):
    height, width = array.shape[:2]
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return array[top:top + crop_size, left:left + crop_size]


def image_to_gray_array(image, crop_to_768=False, crop_size=CROP_SIZE):
    if crop_to_768:
        image = center_crop_pil(image, crop_size)
    image = FF.rgb_to_grayscale(image, 1)
    return np.array(image, dtype=np.float32) / 255.0


def get_patch_sizes(min_patch_size):
    if min_patch_size not in (128, 64, 32):
        raise ValueError('min_patch_size must be one of 128, 64 or 32')
    return [size for size in (128, 64, 32) if size >= min_patch_size]


def get_class_min_patch_sizes(model_class, class_min_patch_sizes=None):
    if class_min_patch_sizes is None:
        class_min_patch_sizes = CLASS_MIN_PATCH_SIZES
    result = {}
    for class_name in model_class:
        if class_name not in class_min_patch_sizes:
            raise ValueError(f'Missing min patch size for class "{class_name}"')
        min_patch_size = class_min_patch_sizes[class_name]
        get_patch_sizes(min_patch_size)
        result[class_name] = min_patch_size
    return result


def parse_class_min_patch_sizes(value):
    if value is None:
        return CLASS_MIN_PATCH_SIZES.copy()
    result = CLASS_MIN_PATCH_SIZES.copy()
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError('class patch sizes must look like "sky=128,car=32"')
        class_name, min_patch_size = item.split('=', 1)
        class_name = class_name.strip()
        min_patch_size = int(min_patch_size.strip())
        get_patch_sizes(min_patch_size)
        result[class_name] = min_patch_size
    return result


def get_class_target(target, class_name):
    return np.isin(target, CITYSCAPES_CLASS_IDS[class_name])


def target_to_color_image(target, model_class):
    color_image = np.zeros((*target.shape, 3), dtype=np.uint8)
    for class_name in model_class:
        color_image[get_class_target(target, class_name)] = CLASS_COLORS[class_name]
    return color_image


def target_to_label(target, model_class):
    label = np.full(target.shape, -1, dtype=np.int64)
    for class_index, class_name in enumerate(model_class):
        label[get_class_target(target, class_name)] = class_index
    return label


def pred_masks_to_label(pred_masks, model_class):
    label = np.full(next(iter(pred_masks.values())).shape, -1, dtype=np.int64)
    for class_index, class_name in enumerate(model_class):
        label[pred_masks[class_name].astype(bool)] = class_index
    return label


def pred_heatmaps_to_label(pred_masks, heatmaps, model_class, class_score_weights):
    shape = next(iter(pred_masks.values())).shape
    label = np.full(shape, -1, dtype=np.int64)
    score_stack = []
    valid_stack = []
    for class_name in model_class:
        score_stack.append(heatmaps[class_name] * class_score_weights.get(class_name, 1.0))
        valid_stack.append(pred_masks[class_name].astype(bool))

    scores = np.stack(score_stack, axis=0)
    valid = np.stack(valid_stack, axis=0)
    masked_scores = np.where(valid, scores, -np.inf)
    has_prediction = valid.any(axis=0)
    best_class = masked_scores.argmax(axis=0)
    label[has_prediction] = best_class[has_prediction]
    return label


def smooth_label_classes(label, score_stack, model_class, smooth_classes,
                         smooth_kernel_size=SMOOTH_KERNEL_SIZE,
                         smooth_remove_isolated_size=SMOOTH_REMOVE_ISOLATED_SIZE):
    smoothed_label = label.copy()
    for class_index, class_name in enumerate(model_class):
        if class_name not in smooth_classes:
            continue
        mask = (label == class_index).astype(np.uint8)
        if not mask.any():
            continue
        smoothed_mask = smooth_mask(
            mask,
            kernel_size=smooth_kernel_size,
            remove_isolated_size=smooth_remove_isolated_size,
        ).astype(bool)
        removed_mask = mask.astype(bool) & ~smoothed_mask
        if removed_mask.any():
            fallback_scores = score_stack.copy()
            fallback_scores[class_index] = -np.inf
            smoothed_label[removed_mask] = fallback_scores.argmax(axis=0)[removed_mask]
        smoothed_label[smoothed_mask] = class_index
    return smoothed_label


def pred_heatmaps_argmax_to_label(heatmaps, model_class, class_score_weights,
                                  use_smooth=True, smooth_classes=SMOOTH_CLASSES,
                                  smooth_kernel_size=SMOOTH_KERNEL_SIZE,
                                  smooth_remove_isolated_size=SMOOTH_REMOVE_ISOLATED_SIZE):
    weighted_scores = [
        heatmaps[class_name] * class_score_weights.get(class_name, 1.0)
        for class_name in model_class
    ]
    score_stack = np.stack(weighted_scores, axis=0)
    label = score_stack.argmax(axis=0).astype(np.int64)
    if use_smooth:
        label = smooth_label_classes(
            label,
            score_stack,
            model_class,
            set(smooth_classes or ()),
            smooth_kernel_size=smooth_kernel_size,
            smooth_remove_isolated_size=smooth_remove_isolated_size,
        )
    return label


def pred_to_label(pred_masks, heatmaps, model_class, merge_mode, class_score_weights,
                  use_smooth=True, smooth_classes=SMOOTH_CLASSES,
                  smooth_kernel_size=SMOOTH_KERNEL_SIZE,
                  smooth_remove_isolated_size=SMOOTH_REMOVE_ISOLATED_SIZE):
    if merge_mode == 'order':
        return pred_masks_to_label(pred_masks, model_class)
    if merge_mode == 'heatmap_score':
        return pred_heatmaps_to_label(pred_masks, heatmaps, model_class, class_score_weights)
    if merge_mode == 'heatmap_argmax':
        return pred_heatmaps_argmax_to_label(
            heatmaps,
            model_class,
            class_score_weights,
            use_smooth=use_smooth,
            smooth_classes=smooth_classes,
            smooth_kernel_size=smooth_kernel_size,
            smooth_remove_isolated_size=smooth_remove_isolated_size,
        )
    raise ValueError('merge_mode must be "order", "heatmap_score" or "heatmap_argmax"')


def make_pred_labels(pred_masks, heatmaps, model_class, class_score_weights,
                     use_smooth=True, smooth_classes=SMOOTH_CLASSES,
                     smooth_kernel_size=SMOOTH_KERNEL_SIZE,
                     smooth_remove_isolated_size=SMOOTH_REMOVE_ISOLATED_SIZE):
    return {
        merge_mode: pred_to_label(
            pred_masks,
            heatmaps,
            model_class,
            merge_mode,
            class_score_weights,
            use_smooth=use_smooth,
            smooth_classes=smooth_classes,
            smooth_kernel_size=smooth_kernel_size,
            smooth_remove_isolated_size=smooth_remove_isolated_size,
        )
        for merge_mode in MERGE_MODES
    }


def label_to_color_image(label, model_class):
    color_image = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_index, class_name in enumerate(model_class):
        color_image[label == class_index] = CLASS_COLORS[class_name]
    return color_image


def compute_pixel_accuracy(pa_counts, model_class):
    pa = pa_counts['correct'] / pa_counts['total'] if pa_counts['total'] > 0 else float('nan')
    class_accuracy = {}
    for class_name in model_class:
        total = pa_counts['class_total'][class_name]
        class_accuracy[class_name] = (
            pa_counts['class_correct'][class_name] / total
            if total > 0
            else float('nan')
        )
    valid_accuracy = [value for value in class_accuracy.values() if not np.isnan(value)]
    mpa = float(np.mean(valid_accuracy)) if valid_accuracy else float('nan')
    return pa, class_accuracy, mpa


def compute_exclusive_iou_from_confusion(confusion, model_class):
    num_classes = len(model_class)
    intersections = np.diag(confusion)
    predicted_pixels = confusion[:, :num_classes].sum(axis=0)
    unions = confusion.sum(axis=1) + predicted_pixels - intersections
    ious = {}
    for class_index, class_name in enumerate(model_class):
        ious[class_name] = (
            float(intersections[class_index] / unions[class_index])
            if unions[class_index] > 0
            else float('nan')
        )
    valid_ious = [value for value in ious.values() if not np.isnan(value)]
    miou = float(np.mean(valid_ious)) if valid_ious else float('nan')
    return ious, miou


def load_models(model_class, device='cuda', min_patch_size=None):
    if min_patch_size is None:
        min_patch_size = CLASS_MIN_PATCH_SIZES[model_class]
    patch_sizes = get_patch_sizes(min_patch_size)
    models = {}
    for patch_size in patch_sizes:
        model = ONN_2class(device=device)
        checkpoint_path = os.path.join(SCRIPT_DIR, 'checkpoints_f', model_class, f'model_{patch_size}.pth')
        print(checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
        model.to(device)
        model.eval()
        models[patch_size] = model
    return models


def parse_mapping(value, default, value_type=float):
    if value is None:
        return default.copy()
    result = default.copy()
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f'Expected comma-separated key=value items, got "{item}"')
        key, raw_value = item.split('=', 1)
        result[key.strip()] = value_type(raw_value.strip())
    return result


def parse_patch_mapping(value, default, value_type=float):
    if value is None:
        return default.copy()
    result = default.copy()
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f'Expected comma-separated patch=value items, got "{item}"')
        patch_size, raw_value = item.split('=', 1)
        result[int(patch_size.strip())] = value_type(raw_value.strip())
    return result


def parse_class_list(value, default):
    if value is None:
        return tuple(default)
    result = []
    for item in value.split(','):
        class_name = item.strip()
        if class_name:
            result.append(class_name)
    return tuple(result)


def get_scan_positions(length, patch_size, stride):
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def pad_min_size(image, min_size):
    height, width = image.shape[:2]
    pad_h = max(0, min_size - height)
    pad_w = max(0, min_size - width)
    if pad_h == 0 and pad_w == 0:
        return image, (height, width)
    return np.pad(image, ((0, pad_h), (0, pad_w)), mode='edge'), (height, width)


def infer_patch_probs(model, patches):
    if not patches:
        return []
    device = next(model.parameters()).device
    batch = np.stack(patches, axis=0)
    batch_tensor = torch.from_numpy(batch).float().unsqueeze(1).to(device, non_blocking=True)
    with torch.inference_mode():
        output = model(batch_tensor)
        batch_probs = torch.softmax(output, dim=1)[:, 1]
    return batch_probs.detach().cpu().numpy().tolist()


def add_patch_size_heatmap(image, model, patch_size, stride, weight, batch_size):
    padded, original_size = pad_min_size(image, patch_size)
    height, width = padded.shape[:2]
    heat_sum = np.zeros((height, width), dtype=np.float32)
    heat_count = np.zeros((height, width), dtype=np.float32)

    positions = [
        (x, y)
        for y in get_scan_positions(height, patch_size, stride)
        for x in get_scan_positions(width, patch_size, stride)
    ]
    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start:start + batch_size]
        patches = [
            padded[y:y + patch_size, x:x + patch_size]
            for x, y in batch_positions
        ]
        probs = infer_patch_probs(model, patches)
        for (x, y), prob in zip(batch_positions, probs):
            heat_sum[y:y + patch_size, x:x + patch_size] += float(prob) * weight
            heat_count[y:y + patch_size, x:x + patch_size] += weight

    original_h, original_w = original_size
    return heat_sum[:original_h, :original_w], heat_count[:original_h, :original_w]


def predict_class_heatmap(image, models, min_patch_size, patch_strides, patch_weights,
                          batch_size=32, normalize_heatmap=True):
    heat_sum = np.zeros_like(image, dtype=np.float32)
    heat_count = np.zeros_like(image, dtype=np.float32)

    for patch_size in get_patch_sizes(min_patch_size):
        if patch_size not in models:
            raise ValueError(f'Missing model for patch size {patch_size}')
        stride = patch_strides.get(patch_size, patch_size)
        weight = patch_weights.get(patch_size, 1.0)
        size_sum, size_count = add_patch_size_heatmap(
            image,
            models[patch_size],
            patch_size,
            stride,
            weight,
            batch_size,
        )
        heat_sum += size_sum
        heat_count += size_count

    if normalize_heatmap:
        heatmap = np.divide(
            heat_sum,
            np.maximum(heat_count, 1e-6),
            out=np.zeros_like(heat_sum, dtype=np.float32),
        )
    else:
        heatmap = heat_sum
    return heatmap


def remove_isolated_blocks(mask, block_size=SMOOTH_REMOVE_ISOLATED_SIZE):
    if block_size is None or block_size <= 1:
        return mask.astype(np.uint8)

    mask = mask.astype(np.uint8)
    filtered = mask.copy()
    min_area = block_size * block_size
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        if area <= min_area or (width <= block_size and height <= block_size):
            filtered[labels == label_index] = 0
    return filtered


def smooth_mask(mask, kernel_size=SMOOTH_KERNEL_SIZE,
                remove_isolated_size=SMOOTH_REMOVE_ISOLATED_SIZE):
    mask = remove_isolated_blocks(mask, block_size=remove_isolated_size).astype(np.float32)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    return cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel).astype(np.uint8)


def predict_heatmap_masks(image, models, model_class, class_min_patch_sizes,
                          heatmap_thresholds, patch_strides, patch_weights,
                          batch_size=32, normalize_heatmap=True, use_smooth=True,
                          smooth_classes=SMOOTH_CLASSES, smooth_kernel_size=SMOOTH_KERNEL_SIZE,
                          smooth_remove_isolated_size=SMOOTH_REMOVE_ISOLATED_SIZE):
    seg_masks = {}
    heatmaps = {}
    smooth_classes = set(smooth_classes or ())
    for class_index, class_name in enumerate(model_class):
        heatmap = predict_class_heatmap(
            image,
            models[class_index],
            class_min_patch_sizes[class_name],
            patch_strides,
            patch_weights,
            batch_size=batch_size,
            normalize_heatmap=normalize_heatmap,
        )
        threshold = heatmap_thresholds[class_name]
        mask = (heatmap >= threshold).astype(np.uint8)
        if use_smooth and class_name in smooth_classes:
            mask = smooth_mask(
                mask,
                kernel_size=smooth_kernel_size,
                remove_isolated_size=smooth_remove_isolated_size,
            )
        seg_masks[class_name] = mask
        heatmaps[class_name] = heatmap
    return seg_masks, heatmaps


def masks_to_color_image(seg_masks, model_class):
    height, width = next(iter(seg_masks.values())).shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    for class_name in model_class:
        color_image[seg_masks[class_name].astype(bool)] = CLASS_COLORS[class_name]
    return color_image


def heatmap_to_uint8(heatmap):
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    if heatmap.max() > heatmap.min():
        normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        normalized = np.zeros_like(heatmap, dtype=np.float32)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def save_heatmap_with_colorbar(path, heatmap, class_name, vmin=HEATMAP_VIS_MIN, vmax=HEATMAP_VIS_MAX):
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    im = ax.imshow(heatmap, cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    ax.set_title(
        f'{class_name} heatmap  min={float(heatmap.min()):.4f}  max={float(heatmap.max()):.4f}',
        fontsize=9,
    )
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label('heatmap score')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def save_heatmap_visualization(image, target, pred_labels, heatmaps, image_path, output_dir,
                               index, model_class, save_heatmaps=True):
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    prefix = f'{index + 1:06d}_{stem}'
    image.save(os.path.join(output_dir, f'{prefix}_image.png'))
    if target is not None:
        Image.fromarray(target_to_color_image(target, model_class)).save(os.path.join(output_dir, f'{prefix}_gt.png'))
    for merge_mode, pred_label in pred_labels.items():
        Image.fromarray(label_to_color_image(pred_label, model_class)).save(
            os.path.join(output_dir, f'{prefix}_pred_{merge_mode}.png')
        )

    if not save_heatmaps:
        return
    heatmap_dir = os.path.join(output_dir, f'{prefix}_heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)
    for class_name in model_class:
        heat_u8 = heatmap_to_uint8(heatmaps[class_name])
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        Image.fromarray(heat_color).save(os.path.join(heatmap_dir, f'{class_name}.png'))
        if SAVE_HEATMAP_COLORBAR:
            save_heatmap_with_colorbar(
                os.path.join(heatmap_dir, f'{class_name}_legend.png'),
                heatmaps[class_name],
                class_name,
            )
        if SAVE_HEATMAP_NUMPY:
            np.save(os.path.join(heatmap_dir, f'{class_name}.npy'), heatmaps[class_name])


def save_class_heatmaps(heatmaps, image_path, output_dir, index, model_class):
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    prefix = f'{index + 1:06d}_{stem}'
    image_heatmap_dir = os.path.join(output_dir, f'{prefix}_heatmaps')
    os.makedirs(image_heatmap_dir, exist_ok=True)

    for class_name in model_class:
        heat_u8 = heatmap_to_uint8(heatmaps[class_name])
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        Image.fromarray(heat_color).save(os.path.join(image_heatmap_dir, f'{class_name}_heatmap.png'))
        Image.fromarray(heat_u8).save(os.path.join(image_heatmap_dir, f'{class_name}_heatmap_gray.png'))
        if SAVE_HEATMAP_COLORBAR:
            save_heatmap_with_colorbar(
                os.path.join(image_heatmap_dir, f'{class_name}_heatmap_legend.png'),
                heatmaps[class_name],
                class_name,
            )
        if SAVE_HEATMAP_NUMPY:
            np.save(os.path.join(image_heatmap_dir, f'{class_name}_heatmap.npy'), heatmaps[class_name])


def update_pixel_accuracy_counts(pa_counts, pred_label, target, model_class):
    target_label = target_to_label(target, model_class)
    valid = target_label >= 0
    pa_counts['correct'] += int((pred_label[valid] == target_label[valid]).sum())
    pa_counts['total'] += int(valid.sum())

    for class_index, class_name in enumerate(model_class):
        class_mask = target_label == class_index
        pa_counts['class_correct'][class_name] += int((pred_label[class_mask] == class_index).sum())
        pa_counts['class_total'][class_name] += int(class_mask.sum())


def update_exclusive_confusion(confusion, pred_label, target, model_class):
    target_label = target_to_label(target, model_class)
    valid = target_label >= 0
    gt = target_label[valid].reshape(-1)
    pred = pred_label[valid].reshape(-1)
    if gt.size:
        num_classes = len(model_class)
        pred = pred.copy()
        pred[pred < 0] = num_classes
        encoded = (num_classes + 1) * gt + pred
        counts = np.bincount(encoded, minlength=num_classes * (num_classes + 1))
        confusion += counts.reshape(num_classes, num_classes + 1)


def compute_dataset_iou(data_root, split, models, model_class, max_images=None, output_json=None,
                        save_vis_num=0, save_vis_dir=None, save_heatmap_num=0, save_heatmap_dir=None,
                        crop_to_768=False, crop_size=CROP_SIZE, class_min_patch_sizes=None,
                        heatmap_thresholds=None, patch_strides=None, patch_weights=None,
                        class_score_weights=None, batch_size=32, normalize_heatmap=True,
                        sort_dataset=False, use_smooth=True, smooth_classes=SMOOTH_CLASSES,
                        smooth_kernel_size=SMOOTH_KERNEL_SIZE,
                        smooth_remove_isolated_size=SMOOTH_REMOVE_ISOLATED_SIZE):
    dataset = Cityscapes(data_root, split=split, target_type='semantic')
    if sort_dataset:
        dataset = sort_cityscapes_dataset(dataset)
    class_min_patch_sizes = get_class_min_patch_sizes(model_class, class_min_patch_sizes)
    heatmap_thresholds = heatmap_thresholds or HEATMAP_THRESHOLDS
    patch_strides = patch_strides or PATCH_STRIDES
    patch_weights = patch_weights or PATCH_WEIGHTS
    class_score_weights = class_score_weights or CLASS_SCORE_WEIGHTS
    smooth_classes = tuple(smooth_classes or ())
    if save_heatmap_dir is None and save_vis_dir is not None:
        save_heatmap_dir = os.path.join(os.path.dirname(save_vis_dir), 'heatmaps')

    intersections = {class_name: 0 for class_name in model_class}
    unions = {class_name: 0 for class_name in model_class}
    pa_counts_by_merge = {
        merge_mode: {
            'correct': 0,
            'total': 0,
            'class_correct': {class_name: 0 for class_name in model_class},
            'class_total': {class_name: 0 for class_name in model_class},
        }
        for merge_mode in MERGE_MODES
    }
    exclusive_confusions = {
        merge_mode: np.zeros((len(model_class), len(model_class) + 1), dtype=np.int64)
        for merge_mode in MERGE_MODES
    }

    total = len(dataset) if max_images is None else min(max_images, len(dataset))
    progress = tqdm(range(total), desc='Evaluate heatmap seg', dynamic_ncols=True)
    for index in progress:
        image, target = dataset[index]
        image_rgb = image
        if crop_to_768:
            image_rgb = center_crop_pil(image_rgb, crop_size)
            target = center_crop_array(target, crop_size)
        image_gray = image_to_gray_array(image_rgb)
        valid = target != 255

        seg_masks, heatmaps = predict_heatmap_masks(
            image_gray,
            models,
            model_class,
            class_min_patch_sizes,
            heatmap_thresholds,
            patch_strides,
            patch_weights,
            batch_size=batch_size,
            normalize_heatmap=normalize_heatmap,
            use_smooth=use_smooth,
            smooth_classes=smooth_classes,
            smooth_kernel_size=smooth_kernel_size,
            smooth_remove_isolated_size=smooth_remove_isolated_size,
        )
        pred_labels = make_pred_labels(
            seg_masks,
            heatmaps,
            model_class,
            class_score_weights,
            use_smooth=use_smooth,
            smooth_classes=smooth_classes,
            smooth_kernel_size=smooth_kernel_size,
            smooth_remove_isolated_size=smooth_remove_isolated_size,
        )

        if save_vis_num and index < save_vis_num:
            save_heatmap_visualization(
                image_rgb,
                target,
                pred_labels,
                heatmaps,
                dataset.images[index],
                save_vis_dir,
                index,
                model_class,
                save_heatmaps=index < save_heatmap_num,
            )
        if save_heatmap_num and index < save_heatmap_num:
            save_class_heatmaps(
                heatmaps,
                dataset.images[index],
                save_heatmap_dir,
                index,
                model_class,
            )

        for class_name in model_class:
            pred = seg_masks[class_name].astype(bool) & valid
            gt = get_class_target(target, class_name) & valid
            intersections[class_name] += int(np.logical_and(pred, gt).sum())
            unions[class_name] += int(np.logical_or(pred, gt).sum())
        for merge_mode, pred_label in pred_labels.items():
            update_pixel_accuracy_counts(pa_counts_by_merge[merge_mode], pred_label, target, model_class)
            update_exclusive_confusion(exclusive_confusions[merge_mode], pred_label, target, model_class)
        progress.set_postfix(image=os.path.basename(dataset.images[index]))

    ious = {
        class_name: (
            intersections[class_name] / unions[class_name]
            if unions[class_name] > 0
            else float('nan')
        )
        for class_name in model_class
    }
    valid_ious = [value for value in ious.values() if not np.isnan(value)]
    miou = float(np.mean(valid_ious)) if valid_ious else float('nan')
    merge_metrics = {}
    for merge_mode in MERGE_MODES:
        exclusive_iou, exclusive_miou = compute_exclusive_iou_from_confusion(
            exclusive_confusions[merge_mode],
            model_class,
        )
        pa, class_accuracy, mpa = compute_pixel_accuracy(pa_counts_by_merge[merge_mode], model_class)
        merge_metrics[merge_mode] = {
            'exclusive_iou': exclusive_iou,
            'exclusive_miou': exclusive_miou,
            'exclusive_confusion_matrix': exclusive_confusions[merge_mode].tolist(),
            'pa': pa,
            'class_accuracy': class_accuracy,
            'mpa': mpa,
            'pa_counts': pa_counts_by_merge[merge_mode],
        }
    metrics_by_method = {
        merge_mode: {
            'iou': merge_metrics[merge_mode]['exclusive_iou'],
            'miou': merge_metrics[merge_mode]['exclusive_miou'],
            'pa': merge_metrics[merge_mode]['pa'],
        }
        for merge_mode in MERGE_MODES
    }

    results = {
        'data_root': data_root,
        'split': split,
        'num_images': total,
        'metrics_by_method': metrics_by_method,
        'heatmap_thresholds': heatmap_thresholds,
        'patch_strides': patch_strides,
        'patch_weights': patch_weights,
        'class_score_weights': class_score_weights,
        'batch_size': batch_size,
        'normalize_heatmap': normalize_heatmap,
        'merge_modes': list(MERGE_MODES),
        'use_smooth': use_smooth,
        'smooth_classes': list(smooth_classes),
        'smooth_kernel_size': smooth_kernel_size,
        'smooth_remove_isolated_size': smooth_remove_isolated_size,
        'save_vis_num': save_vis_num,
        'save_vis_dir': save_vis_dir,
        'save_heatmap_num': save_heatmap_num,
        'save_heatmap_dir': save_heatmap_dir,
        'crop_to_768': crop_to_768,
        'crop_size': crop_size,
        'class_min_patch_sizes': class_min_patch_sizes,
        'sort_dataset': sort_dataset,
    }

    print('\nSegmentation Metrics:')
    for merge_mode, metrics in metrics_by_method.items():
        print(f'\nMethod: {merge_mode}')
        print('IoU:')
        for class_name in model_class:
            print(f'  {class_name}: {metrics["iou"][class_name]:.6f}')
        print(f'mIoU: {metrics["miou"]:.6f}')
        print(f'PA: {metrics["pa"]:.6f}')

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, 'w') as file:
            json.dump(results, file, indent=2)
        print(f'Saved metrics to {output_json}')

    return results


def evaluate_single_image(image_path, models, model_class, result_dir,
                          crop_to_768=False, crop_size=CROP_SIZE, class_min_patch_sizes=None,
                          heatmap_thresholds=None, patch_strides=None, patch_weights=None,
                          class_score_weights=None, batch_size=32, normalize_heatmap=True,
                          use_smooth=True, smooth_classes=SMOOTH_CLASSES,
                          smooth_kernel_size=SMOOTH_KERNEL_SIZE,
                          smooth_remove_isolated_size=SMOOTH_REMOVE_ISOLATED_SIZE):
    image = Image.open(image_path).convert('RGB')
    if crop_to_768:
        image = center_crop_pil(image, crop_size)
    image_gray = image_to_gray_array(image)

    class_min_patch_sizes = get_class_min_patch_sizes(model_class, class_min_patch_sizes)
    heatmap_thresholds = heatmap_thresholds or HEATMAP_THRESHOLDS
    patch_strides = patch_strides or PATCH_STRIDES
    patch_weights = patch_weights or PATCH_WEIGHTS
    class_score_weights = class_score_weights or CLASS_SCORE_WEIGHTS
    smooth_classes = tuple(smooth_classes or ())
    seg_masks, heatmaps = predict_heatmap_masks(
        image_gray,
        models,
        model_class,
        class_min_patch_sizes,
        heatmap_thresholds,
        patch_strides,
        patch_weights,
        batch_size=batch_size,
        normalize_heatmap=normalize_heatmap,
        use_smooth=use_smooth,
        smooth_classes=smooth_classes,
        smooth_kernel_size=smooth_kernel_size,
        smooth_remove_isolated_size=smooth_remove_isolated_size,
    )
    pred_labels = make_pred_labels(
        seg_masks,
        heatmaps,
        model_class,
        class_score_weights,
        use_smooth=use_smooth,
        smooth_classes=smooth_classes,
        smooth_kernel_size=smooth_kernel_size,
        smooth_remove_isolated_size=smooth_remove_isolated_size,
    )

    os.makedirs(result_dir, exist_ok=True)
    image.save(os.path.join(result_dir, 'single_image.png'))
    for merge_mode, pred_label in pred_labels.items():
        Image.fromarray(label_to_color_image(pred_label, model_class)).save(
            os.path.join(result_dir, f'single_pred_{merge_mode}.png')
        )
    heatmap_dir = os.path.join(result_dir, 'single_heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)
    for class_name in model_class:
        heat_u8 = heatmap_to_uint8(heatmaps[class_name])
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        Image.fromarray(heat_color).save(os.path.join(heatmap_dir, f'{class_name}.png'))
        if SAVE_HEATMAP_COLORBAR:
            save_heatmap_with_colorbar(
                os.path.join(heatmap_dir, f'{class_name}_legend.png'),
                heatmaps[class_name],
                class_name,
            )
        if SAVE_HEATMAP_NUMPY:
            np.save(os.path.join(heatmap_dir, f'{class_name}.npy'), heatmaps[class_name])

    return {
        'image_path': image_path,
        'heatmap_thresholds': heatmap_thresholds,
        'patch_strides': patch_strides,
        'patch_weights': patch_weights,
        'class_score_weights': class_score_weights,
        'batch_size': batch_size,
        'normalize_heatmap': normalize_heatmap,
        'merge_modes': list(MERGE_MODES),
        'use_smooth': use_smooth,
        'smooth_classes': list(smooth_classes),
        'smooth_kernel_size': smooth_kernel_size,
        'smooth_remove_isolated_size': smooth_remove_isolated_size,
        'class_min_patch_sizes': class_min_patch_sizes,
        'output_dir': result_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CoPAR heatmap sliding-window segmentation.')
    parser.add_argument('--exp-name', default=EXP_NAME)
    parser.add_argument('--eval-dataset', action='store_true', default=EVAL_DATASET,
                        help='Evaluate a Cityscapes split instead of one image.')
    parser.add_argument('--single-image', action='store_false', dest='eval_dataset',
                        help='Run IMAGE_PATH once even if EVAL_DATASET is True in the file defaults.')
    parser.add_argument('--data-root', default=DATA_ROOT)
    parser.add_argument('--split', default=SPLIT, choices=['train', 'val', 'test'])
    parser.add_argument('--max-images', type=int, default=MAX_IMAGES)
    parser.add_argument('--output-json', default=OUTPUT_JSON)
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--sort-dataset', action='store_true', default=SORT_DATASET)
    parser.add_argument('--no-sort-dataset', action='store_false', dest='sort_dataset')
    parser.add_argument('--save-vis-num', type=int, default=SAVE_VIS_NUM)
    parser.add_argument('--save-vis-dir', default=SAVE_VIS_DIR)
    parser.add_argument('--save-heatmap-num', type=int, default=SAVE_HEATMAP_NUM)
    parser.add_argument('--save-heatmap-dir', default=SAVE_HEATMAP_DIR)
    parser.add_argument('--crop-768', action='store_true', dest='crop_to_768', default=CROP_TO_768)
    parser.add_argument('--no-crop-768', action='store_false', dest='crop_to_768')
    parser.add_argument('--crop-size', type=int, default=CROP_SIZE)
    parser.add_argument('--class-min-patch-sizes', default=None,
                        help='Per-class minimum patch sizes, for example "sky=128,building=64,car=32".')
    parser.add_argument('--heatmap-thresholds', default=None,
                        help='Per-class heatmap thresholds, for example "sky=0.4,car=0.6".')
    parser.add_argument('--patch-strides', default=None,
                        help='Stride per patch size, for example "128=64,64=32,32=16".')
    parser.add_argument('--patch-weights', default=None,
                        help='Weight per patch size when accumulating heatmaps, for example "128=1,64=1,32=2".')
    parser.add_argument('--class-score-weights', default=None,
                        help='Class weights used only by heatmap_score merge, for example "car=1.5,human=2".')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--normalize-heatmap', action='store_true', default=NORMALIZE_HEATMAP)
    parser.add_argument('--no-normalize-heatmap', action='store_false', dest='normalize_heatmap')
    parser.add_argument('--no-smooth', action='store_false', dest='use_smooth', default=USE_SMOOTH,
                        help='Disable smooth post-processing for threshold masks.')
    parser.add_argument('--smooth', action='store_true', dest='use_smooth',
                        help='Enable smooth post-processing for threshold masks.')
    parser.add_argument('--smooth-classes', default=None,
                        help='Comma-separated classes to smooth, for example "car,human".')
    parser.add_argument('--smooth-kernel-size', type=int, default=SMOOTH_KERNEL_SIZE)
    parser.add_argument('--smooth-remove-isolated-size', type=int, default=SMOOTH_REMOVE_ISOLATED_SIZE)
    parser.add_argument('--image-path', default=IMAGE_PATH)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    result_dir = get_result_dir(args.exp_name)
    if args.output_json is None:
        args.output_json = os.path.join(result_dir, 'heatmap_iou_metrics.json')
    if args.save_vis_dir is None:
        args.save_vis_dir = os.path.join(result_dir, 'visualizations')
    if args.save_heatmap_dir is None:
        args.save_heatmap_dir = os.path.join(result_dir, 'heatmaps')

    model_class = MODEL_CLASSES
    class_min_patch_sizes = parse_class_min_patch_sizes(args.class_min_patch_sizes)
    class_min_patch_sizes = get_class_min_patch_sizes(model_class, class_min_patch_sizes)
    heatmap_thresholds = parse_mapping(args.heatmap_thresholds, HEATMAP_THRESHOLDS, value_type=float)
    patch_strides = parse_patch_mapping(args.patch_strides, PATCH_STRIDES, value_type=int)
    patch_weights = parse_patch_mapping(args.patch_weights, PATCH_WEIGHTS, value_type=float)
    class_score_weights = parse_mapping(args.class_score_weights, CLASS_SCORE_WEIGHTS, value_type=float)
    smooth_classes = parse_class_list(args.smooth_classes, SMOOTH_CLASSES)

    models = [
        load_models(class_name, device=args.device, min_patch_size=class_min_patch_sizes[class_name])
        for class_name in model_class
    ]

    if not args.eval_dataset:
        result = evaluate_single_image(
            args.image_path,
            models,
            model_class,
            result_dir,
            crop_to_768=args.crop_to_768,
            crop_size=args.crop_size,
            class_min_patch_sizes=class_min_patch_sizes,
            heatmap_thresholds=heatmap_thresholds,
            patch_strides=patch_strides,
            patch_weights=patch_weights,
            class_score_weights=class_score_weights,
            batch_size=args.batch_size,
            normalize_heatmap=args.normalize_heatmap,
            use_smooth=args.use_smooth,
            smooth_classes=smooth_classes,
            smooth_kernel_size=args.smooth_kernel_size,
            smooth_remove_isolated_size=args.smooth_remove_isolated_size,
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as file:
            json.dump(result, file, indent=2)
        print(f'Saved single-image heatmap result to {result_dir}')
    else:
        if args.data_root is None:
            raise ValueError('Please pass --data-root or set CITYSCAPES_ROOT.')
        compute_dataset_iou(
            args.data_root,
            args.split,
            models,
            model_class,
            max_images=args.max_images,
            output_json=args.output_json,
            save_vis_num=args.save_vis_num,
            save_vis_dir=args.save_vis_dir,
            save_heatmap_num=args.save_heatmap_num,
            save_heatmap_dir=args.save_heatmap_dir,
            crop_to_768=args.crop_to_768,
            crop_size=args.crop_size,
            class_min_patch_sizes=class_min_patch_sizes,
            heatmap_thresholds=heatmap_thresholds,
            patch_strides=patch_strides,
            patch_weights=patch_weights,
            class_score_weights=class_score_weights,
            batch_size=args.batch_size,
            normalize_heatmap=args.normalize_heatmap,
            sort_dataset=args.sort_dataset,
            use_smooth=args.use_smooth,
            smooth_classes=smooth_classes,
            smooth_kernel_size=args.smooth_kernel_size,
            smooth_remove_isolated_size=args.smooth_remove_isolated_size,
        )
