import argparse
import hashlib
import json
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as FF
from PIL import Image
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from copar_evaluate_detect import (
    CLASS_COLORS,
    DETECTION_CLASSES,
    compute_detection_metrics,
    draw_detection_boxes,
    get_depth_path_from_image,
    get_instance_path_from_image,
    get_label_path_from_image,
    load_gt_visualizations,
    quantization_depth,
    save_gray,
    save_rgb,
    to_jsonable,
)
from copar_evaluate_seg_heatmap import (
    get_class_min_patch_sizes as get_heatmap_class_min_patch_sizes,
    label_to_color_image as heatmap_label_to_color_image,
    load_models as load_heatmap_class_models,
    make_pred_labels as make_heatmap_pred_labels,
    predict_heatmap_masks,
    smooth_mask,
)
from utils.cityscapes import Cityscapes


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Edit these defaults directly when running from the IDE.
EXP_NAME = 'watershed_detection'
# False: run IMAGE_PATH/DEPTH_PATH once. True: iterate DATA_ROOT/SPLIT and report metrics.
EVAL_DATASET = True
# False: do not read gtFine at all. True: read GT boxes and report precision/recall/F1.
EVALUATE_GT = True
DATA_ROOT = "data/cityscapes"
GT_ROOT = None
SPLIT = 'train'
MAX_IMAGES = 10
OUTPUT_JSON = None
# In dataset mode, save visualizations for the first SAVE_VIS_NUM images.
SAVE_VIS_NUM = 10
SORT_DATASET = True
# Used only when EVAL_DATASET is False.
IMAGE_PATH = "data/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"
DEPTH_PATH = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CROP_TO_768 = True
CROP_SIZE = 768
USE_SMOOTH = True

HEATMAP_MODEL_CLASSES = ['sky', 'building', 'vegetation', 'road', 'sidewalk', 'traffic', 'car', 'human']
HEATMAP_CLASS_MIN_PATCH_SIZES = {
    'sky': 32,
    'building': 32,
    'vegetation': 32,
    'road': 32,
    'sidewalk': 32,
    'traffic': 32,
    'car': 32,
    'human': 32,
}
HEATMAP_THRESHOLDS = {
    'sky': 0.50,
    'building': 0.50,
    'vegetation': 0.50,
    'road': 0.50,
    'sidewalk': 0.50,
    'traffic': 0.50,
    'car': 0.50,
    'human': 0.50,
}
HEATMAP_CLASS_SCORE_WEIGHTS = {
    'sky': 1.0,
    'building': 1.0,
    'vegetation': 1.0,
    'road': 1.0,
    'sidewalk': 1.0,
    'traffic': 1.0,
    'car': 1.0,
    'human': 1.5,
}
HEATMAP_PATCH_STRIDES = {
    128: 64,
    64: 32,
    32: 16,
}
HEATMAP_PATCH_WEIGHTS = {
    128: 1.0,
    64: 1.0,
    32: 1.0,
}
HEATMAP_BATCH_SIZE = 32
HEATMAP_NORMALIZE = True
HEATMAP_SMOOTH_CLASSES = ('car', 'human', 'vegetation')
HEATMAP_SMOOTH_KERNEL_SIZE = 30
HEATMAP_SMOOTH_REMOVE_ISOLATED_SIZE = 16
HEATMAP_MERGE_MODE = 'heatmap_argmax'
USE_HEATMAP_CACHE = True
REFRESH_HEATMAP_CACHE = False
HEATMAP_CACHE_DIR = None
DEPTH_RANGE_MAX = 51
DEPTH_STRIDE = 1
BOX_MIN_AREA = 64
GT_MIN_AREA = {
    'car': 5000,
    'human': 2000,
}
BOX_THICKNESS = 3
CONFIDENCE_SCORE = 1.0
MATCH_IOU_THRESHOLD = 0.5
BOX_CENTROID_SHIFT_STRENGTH = 0.8
BOX_MAX_CENTROID_SHIFT_RATIO = 0.25
WATERSHED_PARAMS = {
    'car': {
        'min_component_area': 5000,
        'min_instance_area': 2000,
        'min_seed_distance': 40,
        'distance_peak_threshold': 0.25,
        'distance_weight': 0.35,
        'depth_edge_weight': 0.65,
        'depth_marker_threshold': 0.1,
        'min_depth_marker_area': 1000,
        'mean_instance_depth': DEPTH_RANGE_MAX,
        'use_shape_filter': True,
        'min_aspect_ratio': 0.0,
        'max_aspect_ratio': 2.5,
        'min_rectangularity': 0.50,
        'min_solidity': 0.75,
    },
    'human': {
        'min_component_area': 2000,
        'min_instance_area': 500,
        'min_seed_distance': 18,
        'distance_peak_threshold': 0.35,
        'distance_weight': 0.35,
        'depth_edge_weight': 0.65,
        'depth_marker_threshold': 0.1,
        'min_depth_marker_area': 300,
        'mean_instance_depth': DEPTH_RANGE_MAX,
        'use_shape_filter': True,
        'min_aspect_ratio': 0.20,
        'max_aspect_ratio': 1,
        'min_rectangularity': 0.30,
        'min_solidity': 0.30,
    },
}


def get_result_dir(exp_name):
    return os.path.join(SCRIPT_DIR, 'result_center_detect_watershed', exp_name)


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


def load_watershed_image(image_path, depth_path, crop_to_768=True, crop_size=CROP_SIZE):
    image = Image.open(image_path).convert('RGB')
    image_rgb = center_crop_pil(image, crop_size) if crop_to_768 else image
    image_gray = image_to_gray_array(image, crop_to_768=crop_to_768, crop_size=crop_size)

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f'Could not read depth image: {depth_path}')
    depth = depth.astype(np.float32)
    if crop_to_768:
        depth = center_crop_array(depth, crop_size)
    return image_gray, depth, np.array(image_rgb, dtype=np.uint8)


def sort_cityscapes_dataset(dataset):
    pairs = sorted(zip(dataset.images, dataset.targets), key=lambda item: item[0])
    dataset.images = [image for image, _ in pairs]
    dataset.targets = [target for _, target in pairs]
    return dataset


def has_gtfine(data_root, split):
    return os.path.isdir(os.path.join(data_root, 'gtFine', split))


def get_gt_root(args):
    return args.gt_root or args.data_root


def get_gt_lookup_image_path(image_path, args):
    gt_root = get_gt_root(args)
    if gt_root == args.data_root:
        return image_path

    file_name = os.path.basename(image_path)
    city = file_name.split('_')[0]
    return os.path.join(gt_root, 'leftImg8bit', args.split, city, file_name)


def has_gt_for_image(image_path, args):
    gt_image_path = get_gt_lookup_image_path(image_path, args)
    return (
        os.path.exists(get_instance_path_from_image(gt_image_path))
        or os.path.exists(get_label_path_from_image(gt_image_path))
    )


def load_gt_visualizations_for_image(image_path, image_rgb, args, gt_min_area):
    gt_image_path = get_gt_lookup_image_path(image_path, args)
    return load_gt_visualizations(
        gt_image_path,
        image_rgb,
        crop_to_768=args.crop_to_768,
        crop_size=args.crop_size,
        gt_min_area=gt_min_area,
        thickness=args.box_thickness,
    )


def get_gt_min_area(args):
    gt_min_area = GT_MIN_AREA.copy()
    if args.car_gt_min_area is not None:
        gt_min_area['car'] = args.car_gt_min_area
    if args.human_gt_min_area is not None:
        gt_min_area['human'] = args.human_gt_min_area
    return gt_min_area


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


def parse_heatmap_class_min_patch_sizes(value):
    return parse_mapping(value, HEATMAP_CLASS_MIN_PATCH_SIZES, value_type=int)


def get_heatmap_settings(args):
    class_min_patch_sizes = parse_heatmap_class_min_patch_sizes(args.class_min_patch_sizes)
    class_min_patch_sizes = get_heatmap_class_min_patch_sizes(HEATMAP_MODEL_CLASSES, class_min_patch_sizes)
    return {
        'class_min_patch_sizes': class_min_patch_sizes,
        'heatmap_thresholds': parse_mapping(args.heatmap_thresholds, HEATMAP_THRESHOLDS, value_type=float),
        'class_score_weights': parse_mapping(
            args.heatmap_class_score_weights,
            HEATMAP_CLASS_SCORE_WEIGHTS,
            value_type=float,
        ),
        'patch_strides': parse_patch_mapping(args.heatmap_patch_strides, HEATMAP_PATCH_STRIDES, value_type=int),
        'patch_weights': parse_patch_mapping(args.heatmap_patch_weights, HEATMAP_PATCH_WEIGHTS, value_type=float),
        'smooth_classes': parse_class_list(args.heatmap_smooth_classes, HEATMAP_SMOOTH_CLASSES),
    }


def load_segmentation_models(args, settings):
    return [
        load_heatmap_class_models(
            class_name,
            device=args.device,
            min_patch_size=settings['class_min_patch_sizes'][class_name],
        )
        for class_name in HEATMAP_MODEL_CLASSES
    ]


def get_heatmap_cache_key(args, image_path, segmentation_settings):
    image_stat = os.stat(image_path)
    cache_settings = {
        'image_path': os.path.abspath(image_path),
        'image_size': image_stat.st_size,
        'image_mtime_ns': image_stat.st_mtime_ns,
        'model_classes': HEATMAP_MODEL_CLASSES,
        'class_min_patch_sizes': segmentation_settings['class_min_patch_sizes'],
        'patch_strides': segmentation_settings['patch_strides'],
        'patch_weights': segmentation_settings['patch_weights'],
        'batch_size': args.heatmap_batch_size,
        'normalize_heatmap': args.heatmap_normalize,
        'crop_to_768': args.crop_to_768,
        'crop_size': args.crop_size,
    }
    text = json.dumps(cache_settings, sort_keys=True)
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]


def get_heatmap_cache_path(args, image_path, segmentation_settings):
    if not args.use_heatmap_cache:
        return None
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    cache_key = get_heatmap_cache_key(args, image_path, segmentation_settings)
    return os.path.join(args.heatmap_cache_dir, f'{image_id}_{cache_key}.npz')


def load_cached_heatmaps(cache_path):
    if cache_path is None or not os.path.exists(cache_path):
        return None
    with np.load(cache_path) as data:
        if not all(class_name in data for class_name in HEATMAP_MODEL_CLASSES):
            return None
        return {
            class_name: data[class_name].astype(np.float32)
            for class_name in HEATMAP_MODEL_CLASSES
        }


def save_cached_heatmaps(cache_path, heatmaps):
    if cache_path is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez_compressed(
        cache_path,
        **{class_name: heatmaps[class_name].astype(np.float32) for class_name in HEATMAP_MODEL_CLASSES},
    )


def heatmaps_to_masks(args, heatmaps, segmentation_settings):
    seg_masks = {}
    smooth_classes = set(segmentation_settings['smooth_classes'] or ())
    for class_name in HEATMAP_MODEL_CLASSES:
        threshold = segmentation_settings['heatmap_thresholds'][class_name]
        mask = (heatmaps[class_name] >= threshold).astype(np.uint8)
        if args.use_smooth and class_name in smooth_classes:
            mask = smooth_mask(
                mask,
                kernel_size=args.heatmap_smooth_kernel_size,
                remove_isolated_size=args.heatmap_smooth_remove_isolated_size,
            )
        seg_masks[class_name] = mask
    return seg_masks


def heatmap_label_to_masks(label):
    return {
        class_name: (label == class_index).astype(np.uint8)
        for class_index, class_name in enumerate(HEATMAP_MODEL_CLASSES)
    }


def predict_segmentation_masks(args, models_holder, segmentation_settings, image_gray, image_path):
    cache_path = get_heatmap_cache_path(args, image_path, segmentation_settings)
    heatmaps = None
    if args.use_heatmap_cache and not args.refresh_heatmap_cache:
        heatmaps = load_cached_heatmaps(cache_path)
        if heatmaps is not None:
            print(f'Loaded cached heatmaps: {cache_path}')

    if heatmaps is None:
        if models_holder['models'] is None:
            models_holder['models'] = load_segmentation_models(args, segmentation_settings)
        seg_masks, heatmaps = predict_heatmap_masks(
            image_gray,
            models_holder['models'],
            HEATMAP_MODEL_CLASSES,
            segmentation_settings['class_min_patch_sizes'],
            segmentation_settings['heatmap_thresholds'],
            segmentation_settings['patch_strides'],
            segmentation_settings['patch_weights'],
            batch_size=args.heatmap_batch_size,
            normalize_heatmap=args.heatmap_normalize,
            use_smooth=args.use_smooth,
            smooth_classes=segmentation_settings['smooth_classes'],
            smooth_kernel_size=args.heatmap_smooth_kernel_size,
            smooth_remove_isolated_size=args.heatmap_smooth_remove_isolated_size,
        )
        if args.use_heatmap_cache:
            save_cached_heatmaps(cache_path, heatmaps)
            print(f'Saved heatmaps: {cache_path}')
    else:
        seg_masks = heatmaps_to_masks(args, heatmaps, segmentation_settings)

    pred_labels = make_heatmap_pred_labels(
        seg_masks,
        heatmaps,
        HEATMAP_MODEL_CLASSES,
        segmentation_settings['class_score_weights'],
        use_smooth=args.use_smooth,
        smooth_classes=segmentation_settings['smooth_classes'],
        smooth_kernel_size=args.heatmap_smooth_kernel_size,
        smooth_remove_isolated_size=args.heatmap_smooth_remove_isolated_size,
    )
    label = pred_labels[args.heatmap_merge_mode]
    return heatmap_label_to_masks(label), heatmap_label_to_color_image(label, HEATMAP_MODEL_CLASSES)


def list_cityscapes_images(data_root, split, sort_dataset=False):
    images_dir = os.path.join(data_root, 'leftImg8bit', split)
    image_paths = []
    file_names = os.listdir(images_dir)
    direct_images = [file_name for file_name in file_names if file_name.endswith('_leftImg8bit.png')]
    if direct_images:
        if sort_dataset:
            direct_images = sorted(direct_images)
        return [os.path.join(images_dir, file_name) for file_name in direct_images]

    city_names = file_names
    if sort_dataset:
        city_names = sorted(city_names)
    for city in city_names:
        city_dir = os.path.join(images_dir, city)
        if not os.path.isdir(city_dir):
            continue
        file_names = os.listdir(city_dir)
        if sort_dataset:
            file_names = sorted(file_names)
        for file_name in file_names:
            if file_name.endswith('_leftImg8bit.png'):
                image_paths.append(os.path.join(city_dir, file_name))
    return image_paths


def normalize01(array):
    array = array.astype(np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def gradient_magnitude(array):
    array = array.astype(np.float32)
    grad_x = cv2.Sobel(array, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(array, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)


def component_centroid(mask):
    moments = cv2.moments(mask.astype(np.uint8))
    if moments['m00'] == 0:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return np.array([float(xs.mean()), float(ys.mean())])
    return np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']])


def make_markers(distance, component_mask, min_seed_distance, distance_peak_threshold):
    max_distance = float(distance.max())
    if max_distance <= 0:
        return None, 0

    coordinates = peak_local_max(
        distance,
        min_distance=max(1, int(min_seed_distance)),
        threshold_abs=max_distance * float(distance_peak_threshold),
        labels=component_mask.astype(bool),
        exclude_border=False,
    )
    if len(coordinates) == 0:
        y, x = np.unravel_index(np.argmax(distance), distance.shape)
        coordinates = np.array([[y, x]])

    markers = np.zeros(distance.shape, dtype=np.int32)
    for marker_id, (y, x) in enumerate(coordinates, start=1):
        markers[y, x] = marker_id
    return markers, coordinates


def make_depth_markers(depth_edge, component_mask, threshold, min_marker_area):
    if threshold is None or threshold <= 0:
        return None, []

    component_mask = component_mask.astype(np.uint8)
    continuous = ((component_mask > 0) & (depth_edge <= float(threshold))).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(continuous, connectivity=8)

    markers = np.zeros(component_mask.shape, dtype=np.int32)
    coordinates = []
    marker_id = 1
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_marker_area):
            continue
        markers[labels == label] = marker_id
        cx, cy = centroids[label]
        coordinates.append((int(round(cy)), int(round(cx))))
        marker_id += 1

    if marker_id <= 2:
        return None, []
    return markers, np.array(coordinates, dtype=np.int64)


def filter_class_components(class_mask, params):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask.astype(np.uint8), connectivity=8)
    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < params['min_component_area']:
            continue
        components.append((labels == label).astype(np.uint8))
    return components


def split_component_by_depth_watershed(component_mask, depth_q, params):
    component_mask = component_mask.astype(np.uint8)
    distance = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
    distance = cv2.GaussianBlur(distance, (0, 0), sigmaX=1.0)
    depth_edge = normalize01(gradient_magnitude(depth_q))
    markers, seed_coordinates = make_depth_markers(
        depth_edge,
        component_mask,
        params.get('depth_marker_threshold', 0.0),
        params.get('min_depth_marker_area', params['min_instance_area']),
    )
    if markers is None:
        markers, seed_coordinates = make_markers(
            distance,
            component_mask,
            params['min_seed_distance'],
            params['distance_peak_threshold'],
        )
    if markers is None:
        return [], []

    seeds = [(int(x), int(y)) for y, x in seed_coordinates]
    if len(seed_coordinates) <= 1:
        return ([component_mask], seeds) if int(component_mask.sum()) >= params['min_instance_area'] else ([], seeds)

    distance_term = 1.0 - normalize01(distance)
    elevation = (
        float(params.get('distance_weight', 1.0)) * distance_term
        + float(params['depth_edge_weight']) * depth_edge
    )
    labels = watershed(elevation, markers=markers, mask=component_mask.astype(bool))

    instances = []
    for label in range(1, labels.max() + 1):
        instance_mask = (labels == label).astype(np.uint8)
        if int(instance_mask.sum()) >= params['min_instance_area']:
            instances.append(instance_mask)
    return instances, seeds


def is_instance_shape_valid(mask, params):
    if not params.get('use_shape_filter', True):
        return True

    mask = mask.astype(np.uint8)
    area = int(mask.sum())
    if area <= 0:
        return False

    x, y, width, height = cv2.boundingRect(mask)
    if width <= 0 or height <= 0:
        return False

    aspect_ratio = width / height
    rectangularity = area / float(width * height)
    if aspect_ratio < float(params['min_aspect_ratio']):
        return False
    if aspect_ratio > float(params['max_aspect_ratio']):
        return False
    if rectangularity < float(params['min_rectangularity']):
        return False

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0:
        return False
    solidity = area / hull_area
    return solidity >= float(params['min_solidity'])


def disparity_to_metric_depth(depth):
    disp = depth.astype(np.float32) / 256.0
    return (2262.52 * 0.209) / (disp + 1e-6)


def is_instance_depth_valid(mask, depth, params):
    mean_instance_depth = params.get('mean_instance_depth')
    if mean_instance_depth is None or mean_instance_depth <= 0:
        return True

    valid = (mask > 0) & (depth > 0)
    if not valid.any():
        return False

    metric_depth = disparity_to_metric_depth(depth)
    return float(metric_depth[valid].mean()) <= float(mean_instance_depth)


def get_instances_by_watershed(depth_q, depth, class_mask, params):
    instances = []
    seed_points = []
    for component_mask in filter_class_components(class_mask, params):
        instance_masks, component_seeds = split_component_by_depth_watershed(component_mask, depth_q, params)
        seed_points.extend(component_seeds)
        for instance_mask in instance_masks:
            instance_mask = instance_mask.astype(np.uint8)
            if not is_instance_depth_valid(instance_mask, depth, params):
                continue
            if not is_instance_shape_valid(instance_mask, params):
                continue
            centroid = component_centroid(instance_mask)
            if centroid is not None:
                instances.append((instance_mask, centroid))
    return instances, seed_points


def get_watershed_params(args):
    params = {
        class_name: class_params.copy()
        for class_name, class_params in WATERSHED_PARAMS.items()
    }
    overrides_by_class = {
        'car': {
            'min_component_area': args.car_min_component_area,
            'min_instance_area': args.car_min_instance_area,
            'min_seed_distance': args.car_min_seed_distance,
            'distance_peak_threshold': args.car_distance_peak_threshold,
            'distance_weight': args.car_distance_weight,
            'depth_edge_weight': args.car_depth_edge_weight,
            'depth_marker_threshold': args.car_depth_marker_threshold,
            'min_depth_marker_area': args.car_min_depth_marker_area,
            'mean_instance_depth': args.car_mean_instance_depth,
            'min_aspect_ratio': args.car_min_aspect_ratio,
            'max_aspect_ratio': args.car_max_aspect_ratio,
            'min_rectangularity': args.car_min_rectangularity,
            'min_solidity': args.car_min_solidity,
        },
        'human': {
            'min_component_area': args.human_min_component_area,
            'min_instance_area': args.human_min_instance_area,
            'min_seed_distance': args.human_min_seed_distance,
            'distance_peak_threshold': args.human_distance_peak_threshold,
            'distance_weight': args.human_distance_weight,
            'depth_edge_weight': args.human_depth_edge_weight,
            'depth_marker_threshold': args.human_depth_marker_threshold,
            'min_depth_marker_area': args.human_min_depth_marker_area,
            'mean_instance_depth': args.human_mean_instance_depth,
            'min_aspect_ratio': args.human_min_aspect_ratio,
            'max_aspect_ratio': args.human_max_aspect_ratio,
            'min_rectangularity': args.human_min_rectangularity,
            'min_solidity': args.human_min_solidity,
        },
    }
    for class_name, overrides in overrides_by_class.items():
        for key, value in overrides.items():
            if value is not None:
                params[class_name][key] = value
    if args.no_car_shape_filter:
        params['car']['use_shape_filter'] = False
    if args.no_human_shape_filter:
        params['human']['use_shape_filter'] = False
    return params


def make_detection_outputs(image_rgb, depth_q, masks_and_centers_by_class):
    color_list = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 128, 0), (0, 128, 128),
    ]
    result1 = image_rgb.copy()
    result2 = image_rgb.copy()
    object_masks = []

    color_index = 0
    for class_name in ('car', 'human'):
        radius = 8 if class_name == 'car' else 6
        for mask, center in masks_and_centers_by_class.get(class_name, []):
            color = color_list[color_index % len(color_list)]
            color_index += 1
            cv2.circle(result1, (int(center[0]), int(center[1])), radius, CLASS_COLORS[class_name], -1)
            result2[mask > 0] = color
            object_masks.append((class_name, mask.astype(np.uint8), color))

    return result1, result2, depth_q, object_masks


def draw_shape_filter_comparison(image_rgb, before_by_class, after_by_class):
    """Visualize instance masks before and after use_shape_filter.

    Left: instances generated with use_shape_filter disabled.
    Right: instances kept after the current shape filter parameters.
    """
    color_list = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 128, 0), (0, 128, 128),
    ]

    def overlay_instances(title, masks_and_centers_by_class):
        canvas = image_rgb.copy()
        color_index = 0
        counts = {}
        for class_name in ('car', 'human'):
            items = masks_and_centers_by_class.get(class_name, [])
            counts[class_name] = len(items)
            for mask, center in items:
                color = color_list[color_index % len(color_list)]
                color_index += 1
                canvas[mask > 0] = color
                cv2.circle(canvas, (int(center[0]), int(center[1])), 5, CLASS_COLORS[class_name], -1)
        text = f'{title} | car={counts.get("car", 0)}, human={counts.get("human", 0)}'
        cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        return canvas

    before = overlay_instances('Before shape filter', before_by_class)
    after = overlay_instances('After shape filter', after_by_class)
    return np.concatenate([before, after], axis=1)


def clip_bbox_to_image(bbox, image_shape):
    height, width = image_shape
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(0, min(width - 1, int(round(x2)))),
        max(0, min(height - 1, int(round(y2)))),
    ]


def bbox_from_center_size(center_x, center_y, box_width, box_height, image_shape):
    height, width = image_shape
    box_width = max(1.0, float(box_width))
    box_height = max(1.0, float(box_height))

    x1 = center_x - (box_width - 1.0) / 2.0
    x2 = center_x + (box_width - 1.0) / 2.0
    y1 = center_y - (box_height - 1.0) / 2.0
    y2 = center_y + (box_height - 1.0) / 2.0

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 >= width:
        overflow = x2 - (width - 1)
        x1 -= overflow
        x2 = width - 1
    if y2 >= height:
        overflow = y2 - (height - 1)
        y1 -= overflow
        y2 = height - 1

    return clip_bbox_to_image([x1, y1, x2, y2], image_shape)


def adjust_bbox_by_mask_centroid(bbox, mask, image_shape,
                                 shift_strength=BOX_CENTROID_SHIFT_STRENGTH,
                                 max_shift_ratio=BOX_MAX_CENTROID_SHIFT_RATIO):
    if bbox is None:
        return None, None

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return bbox, None

    x1, y1, x2, y2 = bbox
    box_width = max(1, x2 - x1 + 1)
    box_height = max(1, y2 - y1 + 1)
    box_center_x = (x1 + x2) / 2.0
    box_center_y = (y1 + y2) / 2.0
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())

    if shift_strength is None or shift_strength <= 0:
        shift_x = 0
        shift_y = 0
    else:
        raw_shift_x = float(shift_strength) * (centroid_x - box_center_x)
        raw_shift_y = float(shift_strength) * (centroid_y - box_center_y)
        max_shift_x = max(0.0, float(max_shift_ratio) * box_width)
        max_shift_y = max(0.0, float(max_shift_ratio) * box_height)
        shift_x = int(round(np.clip(raw_shift_x, -max_shift_x, max_shift_x)))
        shift_y = int(round(np.clip(raw_shift_y, -max_shift_y, max_shift_y)))

    shifted_bbox = clip_bbox_to_image(
        [x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y],
        image_shape,
    )
    adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2 = shifted_bbox
    return shifted_bbox, {
        'bbox_source': 'full_mask',
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
        'box_center_x_before_shift': box_center_x,
        'box_center_y_before_shift': box_center_y,
        'requested_shift_x': shift_x,
        'requested_shift_y': shift_y,
        'actual_shift_x': adjusted_x1 - x1,
        'actual_shift_y': adjusted_y1 - y1,
        'full_bbox_before_adjustment': [int(x1), int(y1), int(x2), int(y2)],
    }


def bbox_from_mask(mask, image_shape,
                   centroid_shift_strength=BOX_CENTROID_SHIFT_STRENGTH,
                   max_centroid_shift_ratio=BOX_MAX_CENTROID_SHIFT_RATIO):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None

    full_box = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    ]
    return adjust_bbox_by_mask_centroid(
        full_box,
        mask,
        image_shape,
        shift_strength=centroid_shift_strength,
        max_shift_ratio=max_centroid_shift_ratio,
    )


def masks_to_robust_detections(object_masks, image_shape, min_area=64,
                               confidence_score=1.0,
                               centroid_shift_strength=BOX_CENTROID_SHIFT_STRENGTH,
                               max_centroid_shift_ratio=BOX_MAX_CENTROID_SHIFT_RATIO):
    detections = []

    for class_name, mask, _ in object_masks:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            component_mask = (labels == label).astype(np.uint8)
            bbox, bbox_adjustment = bbox_from_mask(
                component_mask,
                image_shape,
                centroid_shift_strength=centroid_shift_strength,
                max_centroid_shift_ratio=max_centroid_shift_ratio,
            )
            if bbox is None:
                continue

            detections.append({
                'class': class_name,
                'bbox_xyxy': bbox,
                'area': area,
                'score': confidence_score,
                'bbox_adjustment': bbox_adjustment,
            })
    return detections

def draw_seed_points(image_rgb, seed_points_by_class):
    seeds_image = image_rgb.copy()
    for class_name, seed_points in seed_points_by_class.items():
        radius = 6 if class_name == 'car' else 5
        for x, y in seed_points:
            cv2.circle(seeds_image, (int(x), int(y)), radius, CLASS_COLORS[class_name], -1)
            cv2.circle(seeds_image, (int(x), int(y)), radius + 2, (255, 255, 255), 1)
    return seeds_image


def predict_image_detections(args, models_holder, segmentation_settings, image_path, depth_path):
    image_gray, depth, image_rgb = load_watershed_image(
        image_path,
        depth_path,
        crop_to_768=args.crop_to_768,
        crop_size=args.crop_size,
    )
    seg_masks, segmentation = predict_segmentation_masks(
        args,
        models_holder,
        segmentation_settings,
        image_gray,
        image_path,
    )

    depth_q = quantization_depth(depth, args.depth_range_max, max(1, args.depth_stride // 2))
    if depth_q.shape != seg_masks['car'].shape:
        height, width = depth_q.shape
        seg_masks = {key: mask[:height, :width] for key, mask in seg_masks.items()}
        segmentation = segmentation[:height, :width]
        image_rgb = image_rgb[:height, :width]
        depth = depth[:height, :width]

    watershed_params = get_watershed_params(args)
    masks_and_centers_by_class = {}
    masks_before_shape_filter_by_class = {}
    seed_points_by_class = {}
    for class_name in ('car', 'human'):
        # First generate instances with use_shape_filter disabled. This is the
        # "before" side of the comparison. Depth filtering is still applied.
        params_before_shape_filter = watershed_params[class_name].copy()
        params_before_shape_filter['use_shape_filter'] = False
        masks_before_shape_filter, seed_points = get_instances_by_watershed(
            depth_q,
            depth,
            seg_masks[class_name],
            params_before_shape_filter,
        )

        # Then apply the current shape filter parameters to get the "after" side.
        masks_after_shape_filter = [
            (instance_mask, centroid)
            for instance_mask, centroid in masks_before_shape_filter
            if is_instance_shape_valid(instance_mask, watershed_params[class_name])
        ]

        masks_before_shape_filter_by_class[class_name] = masks_before_shape_filter
        masks_and_centers_by_class[class_name] = masks_after_shape_filter
        seed_points_by_class[class_name] = seed_points

    shape_filter_comparison = draw_shape_filter_comparison(
        image_rgb,
        masks_before_shape_filter_by_class,
        masks_and_centers_by_class,
    )
    seeds = draw_seed_points(image_rgb, seed_points_by_class)
    result1, result2, result3, object_masks = make_detection_outputs(
        image_rgb,
        depth_q,
        masks_and_centers_by_class,
    )
    detections = masks_to_robust_detections(
        object_masks,
        image_rgb.shape[:2],
        min_area=args.box_min_area,
        confidence_score=args.confidence_score,
        centroid_shift_strength=args.box_centroid_shift_strength,
        max_centroid_shift_ratio=args.box_max_centroid_shift_ratio,
    )
    detection = draw_detection_boxes(image_rgb, detections, thickness=args.box_thickness)
    return image_rgb, segmentation, seeds, result1, result2, result3, detection, detections, shape_filter_comparison


def run_single_image(args):
    result_dir = get_result_dir(args.exp_name)
    os.makedirs(result_dir, exist_ok=True)
    if args.output_json is None:
        args.output_json = os.path.join(result_dir, 'detection_results.json')
    if args.heatmap_cache_dir is None:
        args.heatmap_cache_dir = os.path.join(result_dir, 'heatmap_cache')
    depth_path = args.depth_path or get_depth_path_from_image(args.image_path)

    segmentation_settings = get_heatmap_settings(args)
    models_holder = {'models': None}
    image_rgb, segmentation, seeds, result1, result2, result3, detection, detections, shape_filter_comparison = predict_image_detections(
        args,
        models_holder,
        segmentation_settings,
        args.image_path,
        depth_path,
    )
    has_gt = args.evaluate_gt and has_gt_for_image(args.image_path, args)
    gt_boxes = None
    gt_min_area = get_gt_min_area(args)
    if has_gt:
        gt_instance, gt_detection, gt_boxes = load_gt_visualizations_for_image(
            args.image_path,
            image_rgb,
            args,
            gt_min_area,
        )

    save_rgb(os.path.join(result_dir, 'segmentation.png'), segmentation)
    save_rgb(os.path.join(result_dir, 'seeds.png'), seeds)
    save_rgb(os.path.join(result_dir, 'result1.png'), result1)
    save_rgb(os.path.join(result_dir, 'result2.png'), result2)
    save_gray(os.path.join(result_dir, 'result3.png'), result3)
    save_rgb(os.path.join(result_dir, 'detection.png'), detection)
    save_rgb(os.path.join(result_dir, 'shape_filter_comparison.png'), shape_filter_comparison)
    if has_gt:
        save_rgb(os.path.join(result_dir, 'gt_instance.png'), gt_instance)
        save_rgb(os.path.join(result_dir, 'gt_detection.png'), gt_detection)

    image_id = os.path.splitext(os.path.basename(args.image_path))[0]
    metric_predictions = []
    gt_by_image = {}
    metrics_per_class = {
        class_name: {
            'precision': float('nan'),
            'recall': float('nan'),
            'f1': float('nan'),
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'num_gt': 0,
        }
        for class_name in DETECTION_CLASSES
    }
    overall_metrics = {
        'precision': float('nan'),
        'recall': float('nan'),
        'f1': float('nan'),
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'num_gt': 0,
    }
    num_gt = {class_name: 0 for class_name in DETECTION_CLASSES}
    if has_gt:
        for det in detections:
            det = det.copy()
            det['image_id'] = image_id
            metric_predictions.append(det)
        gt_by_image[image_id] = gt_boxes
        metrics_per_class, overall_metrics, num_gt = compute_detection_metrics(
            metric_predictions,
            gt_by_image,
            args.match_iou_threshold,
        )

    results = {
        'parameters': {
            'exp_name': args.exp_name,
            'mode': 'single_image',
            'method': 'watershed',
            'evaluate_gt': args.evaluate_gt,
            'gt_root': get_gt_root(args),
            'match_iou_threshold': args.match_iou_threshold,
            'confidence_score': args.confidence_score,
            'gt_min_area': gt_min_area,
            'box_min_area': args.box_min_area,
            'bbox_method': 'full_mask_bbox_plus_centroid_shift',
            'box_centroid_shift_strength': args.box_centroid_shift_strength,
            'box_max_centroid_shift_ratio': args.box_max_centroid_shift_ratio,
            'watershed_params': get_watershed_params(args),
            'segmentation_method': args.heatmap_merge_mode,
            'segmentation_settings': segmentation_settings,
            'heatmap_cache_dir': args.heatmap_cache_dir,
            'use_heatmap_cache': args.use_heatmap_cache,
            'refresh_heatmap_cache': args.refresh_heatmap_cache,
            'crop_to_768': args.crop_to_768,
            'crop_size': args.crop_size,
            'use_smooth': args.use_smooth,
            'depth_range_max': args.depth_range_max,
            'depth_stride': args.depth_stride,
        },
        'metrics': {
            'overall': overall_metrics,
            'per_class': metrics_per_class,
            'precision': overall_metrics['precision'],
            'recall': overall_metrics['recall'],
            'f1': overall_metrics['f1'],
            'num_gt': num_gt,
        },
        'image': {
            'image_path': args.image_path,
            'depth_path': depth_path,
            'has_gt': has_gt,
        },
        'detections': detections,
        'gt_boxes': gt_boxes,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w') as file:
        json.dump(to_jsonable(results), file, indent=2)

    print(f'Saved watershed result1/result2/result3/detection to {result_dir}')
    print(f'Saved detections to {args.output_json}')
    print(f"Overall precision/recall/F1@{args.match_iou_threshold}: "
          f"{overall_metrics['precision']:.6f}/"
          f"{overall_metrics['recall']:.6f}/"
          f"{overall_metrics['f1']:.6f}")


def save_detection_visualizations(result_dir, image_id, segmentation, seeds, result1, result2, result3,
                                  detection, shape_filter_comparison, gt_instance, gt_detection):
    vis_dir = os.path.join(result_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    save_rgb(os.path.join(vis_dir, f'{image_id}_segmentation.png'), segmentation)
    save_rgb(os.path.join(vis_dir, f'{image_id}_seeds.png'), seeds)
    save_rgb(os.path.join(vis_dir, f'{image_id}_result1.png'), result1)
    save_rgb(os.path.join(vis_dir, f'{image_id}_result2.png'), result2)
    save_gray(os.path.join(vis_dir, f'{image_id}_result3.png'), result3)
    save_rgb(os.path.join(vis_dir, f'{image_id}_detection.png'), detection)
    save_rgb(os.path.join(vis_dir, f'{image_id}_shape_filter_comparison.png'), shape_filter_comparison)
    if gt_instance is not None:
        save_rgb(os.path.join(vis_dir, f'{image_id}_gt_instance.png'), gt_instance)
    if gt_detection is not None:
        save_rgb(os.path.join(vis_dir, f'{image_id}_gt_detection.png'), gt_detection)


def run_dataset(args):
    result_dir = get_result_dir(args.exp_name)
    os.makedirs(result_dir, exist_ok=True)
    if args.output_json is None:
        args.output_json = os.path.join(result_dir, 'precision_recall_f1_metrics.json')
    if args.heatmap_cache_dir is None:
        args.heatmap_cache_dir = os.path.join(result_dir, 'heatmap_cache')

    gt_root = get_gt_root(args)
    evaluate_gt = args.evaluate_gt and has_gtfine(gt_root, args.split)
    if args.evaluate_gt and not evaluate_gt:
        print(f'gtFine/{args.split} not found under {gt_root}; running without GT metrics.')

    image_paths = list_cityscapes_images(args.data_root, args.split, sort_dataset=args.sort_dataset)
    total = len(image_paths) if args.max_images is None else min(args.max_images, len(image_paths))
    segmentation_settings = get_heatmap_settings(args)
    models_holder = {'models': None}

    predictions = []
    gt_by_image = {}
    gt_min_area = get_gt_min_area(args)

    for index in range(total):
        image_path = image_paths[index]
        depth_path = get_depth_path_from_image(image_path)
        image_id = os.path.splitext(os.path.basename(image_path))[0]

        image_rgb, segmentation, seeds, result1, result2, result3, detection, detections, shape_filter_comparison = predict_image_detections(
            args,
            models_holder,
            segmentation_settings,
            image_path,
            depth_path,
        )

        for det in detections:
            det = det.copy()
            det['image_id'] = image_id
            predictions.append(det)

        gt_instance = None
        gt_detection = None
        if evaluate_gt:
            gt_instance, gt_detection, gt_boxes = load_gt_visualizations_for_image(
                image_path,
                image_rgb,
                args,
                gt_min_area,
            )
            gt_by_image[image_id] = gt_boxes

        if args.save_vis_num and index < args.save_vis_num:
            save_detection_visualizations(
                result_dir,
                image_id,
                segmentation,
                seeds,
                result1,
                result2,
                result3,
                detection,
                shape_filter_comparison,
                gt_instance,
                gt_detection,
            )

        print(f'[{index + 1}/{total}] {os.path.basename(image_path)} detections={len(detections)}')

    if evaluate_gt:
        metrics_per_class, overall_metrics, num_gt = compute_detection_metrics(
            predictions,
            gt_by_image,
            args.match_iou_threshold,
        )
    else:
        metrics_per_class = {
            class_name: {
                'precision': float('nan'),
                'recall': float('nan'),
                'f1': float('nan'),
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'num_gt': 0,
            }
            for class_name in DETECTION_CLASSES
        }
        overall_metrics = {
            'precision': float('nan'),
            'recall': float('nan'),
            'f1': float('nan'),
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'num_gt': 0,
        }
        num_gt = {class_name: 0 for class_name in DETECTION_CLASSES}
    metrics = {
        'parameters': {
            'exp_name': args.exp_name,
            'mode': 'dataset',
            'method': 'watershed',
            'evaluate_gt': evaluate_gt,
            'requested_evaluate_gt': args.evaluate_gt,
            'gt_root': gt_root,
            'data_root': args.data_root,
            'split': args.split,
            'max_images': args.max_images,
            'sort_dataset': args.sort_dataset,
            'match_iou_threshold': args.match_iou_threshold,
            'confidence_score': args.confidence_score,
            'gt_min_area': gt_min_area,
            'box_min_area': args.box_min_area,
            'bbox_method': 'full_mask_bbox_plus_centroid_shift',
            'box_centroid_shift_strength': args.box_centroid_shift_strength,
            'box_max_centroid_shift_ratio': args.box_max_centroid_shift_ratio,
            'watershed_params': get_watershed_params(args),
            'segmentation_method': args.heatmap_merge_mode,
            'segmentation_settings': segmentation_settings,
            'heatmap_cache_dir': args.heatmap_cache_dir,
            'use_heatmap_cache': args.use_heatmap_cache,
            'refresh_heatmap_cache': args.refresh_heatmap_cache,
            'crop_to_768': args.crop_to_768,
            'crop_size': args.crop_size,
            'use_smooth': args.use_smooth,
            'depth_range_max': args.depth_range_max,
            'depth_stride': args.depth_stride,
        },
        'metrics': {
            'overall': overall_metrics,
            'per_class': metrics_per_class,
            'precision': overall_metrics['precision'],
            'recall': overall_metrics['recall'],
            'f1': overall_metrics['f1'],
            'num_gt': num_gt,
            'num_predictions': len(predictions),
        },
        'dataset': {
            'num_images': total,
        },
        'detections': predictions,
        'gt_boxes': gt_by_image,
    }

    print('\nDetection Metrics:')
    if evaluate_gt:
        for class_name, values in metrics_per_class.items():
            print(
                f"  {class_name}: precision={values['precision']:.6f}, "
                f"recall={values['recall']:.6f}, f1={values['f1']:.6f}, "
                f"tp={values['tp']}, fp={values['fp']}, fn={values['fn']}"
            )
        print(
            f"Overall@{args.match_iou_threshold}: "
            f"precision={overall_metrics['precision']:.6f}, "
            f"recall={overall_metrics['recall']:.6f}, "
            f"f1={overall_metrics['f1']:.6f}"
        )
    else:
        print('  evaluate_gt=False, skipped GT metrics.')

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w') as file:
        json.dump(to_jsonable(metrics), file, indent=2)
    print(f'Saved metrics to {args.output_json}')


def parse_args():
    parser = argparse.ArgumentParser(description='Generate CoPAR watershed detection boxes.')
    parser.add_argument('--exp-name', default=EXP_NAME)
    parser.add_argument('--eval-dataset', action='store_true', default=EVAL_DATASET,
                        help='Evaluate precision/recall/F1 on a Cityscapes split instead of one image.')
    parser.add_argument('--single-image', action='store_false', dest='eval_dataset',
                        help='Run one image even if EVAL_DATASET is True in the file defaults.')
    parser.add_argument('--evaluate-gt', action='store_true', default=EVALUATE_GT,
                        help='Read gtFine and compute precision/recall/F1.')
    parser.add_argument('--no-evaluate-gt', action='store_false', dest='evaluate_gt',
                        help='Do not read gtFine; only save predictions.')
    parser.add_argument('--data-root', default=DATA_ROOT,
                        help='Cityscapes root containing leftImg8bit/, gtFine/ and disparity/.')
    parser.add_argument('--gt-root', default=GT_ROOT,
                        help='Optional Cityscapes root containing gtFine/. Use this when DATA_ROOT is an exported RGB/depth folder.')
    parser.add_argument('--split', default=SPLIT, choices=['train', 'val', 'test'])
    parser.add_argument('--max-images', type=int, default=MAX_IMAGES)
    parser.add_argument('--output-json', default=OUTPUT_JSON)
    parser.add_argument('--save-vis-num', type=int, default=SAVE_VIS_NUM)
    parser.add_argument('--sort-dataset', action='store_true', default=SORT_DATASET,
                        help='Sort dataset images by path before taking --max-images.')
    parser.add_argument('--no-sort-dataset', action='store_false', dest='sort_dataset',
                        help='Use filesystem/Cityscapes dataset order.')
    parser.add_argument('--image-path', default=IMAGE_PATH)
    parser.add_argument('--depth-path', default=DEPTH_PATH)
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--crop-768', action='store_true', dest='crop_to_768', default=CROP_TO_768)
    parser.add_argument('--no-crop-768', action='store_false', dest='crop_to_768')
    parser.add_argument('--crop-size', type=int, default=CROP_SIZE)
    parser.add_argument('--no-smooth', action='store_false', dest='use_smooth', default=USE_SMOOTH)
    parser.add_argument('--heatmap-merge-mode', default=HEATMAP_MERGE_MODE,
                        choices=['heatmap_score', 'heatmap_argmax'],
                        help='Use thresholded score merge or full heatmap argmax segmentation before watershed.')
    parser.add_argument('--class-min-patch-sizes')
    parser.add_argument('--heatmap-thresholds',
                        help='Per-class heatmap thresholds, for example "car=0.4,human=0.3".')
    parser.add_argument('--heatmap-class-score-weights',
                        help='Per-class score weights, for example "car=1.0,human=1.5".')
    parser.add_argument('--heatmap-patch-strides',
                        help='Stride per patch size, for example "128=64,64=32,32=16".')
    parser.add_argument('--heatmap-patch-weights',
                        help='Weight per patch size, for example "128=1,64=1,32=2".')
    parser.add_argument('--heatmap-batch-size', type=int, default=HEATMAP_BATCH_SIZE)
    parser.add_argument('--heatmap-normalize', action='store_true', default=HEATMAP_NORMALIZE)
    parser.add_argument('--no-heatmap-normalize', action='store_false', dest='heatmap_normalize')
    parser.add_argument('--heatmap-smooth-classes',
                        help='Comma-separated classes smoothed in heatmap segmentation, e.g. "car,human".')
    parser.add_argument('--heatmap-smooth-kernel-size', type=int, default=HEATMAP_SMOOTH_KERNEL_SIZE)
    parser.add_argument('--heatmap-smooth-remove-isolated-size', type=int,
                        default=HEATMAP_SMOOTH_REMOVE_ISOLATED_SIZE)
    parser.add_argument('--heatmap-cache-dir', default=HEATMAP_CACHE_DIR)
    parser.add_argument('--use-heatmap-cache', action='store_true', default=USE_HEATMAP_CACHE)
    parser.add_argument('--no-heatmap-cache', action='store_false', dest='use_heatmap_cache')
    parser.add_argument('--refresh-heatmap-cache', action='store_true', default=REFRESH_HEATMAP_CACHE,
                        help='Ignore existing cached heatmaps and overwrite them after inference.')
    parser.add_argument('--depth-range-max', type=int, default=DEPTH_RANGE_MAX)
    parser.add_argument('--depth-stride', type=int, default=DEPTH_STRIDE)
    parser.add_argument('--box-min-area', type=int, default=BOX_MIN_AREA)
    parser.add_argument('--box-centroid-shift-strength', type=float, default=BOX_CENTROID_SHIFT_STRENGTH,
                        help='Move the final bbox toward the mask centroid by this fraction of the center offset. Use 0 to disable.')
    parser.add_argument('--box-max-centroid-shift-ratio', type=float, default=BOX_MAX_CENTROID_SHIFT_RATIO,
                        help='Maximum bbox shift as a fraction of bbox width/height.')
    parser.add_argument('--car-gt-min-area', type=int, default=None)
    parser.add_argument('--human-gt-min-area', type=int, default=None)
    parser.add_argument('--box-thickness', type=int, default=BOX_THICKNESS)
    parser.add_argument('--confidence-score', type=float, default=CONFIDENCE_SCORE)
    parser.add_argument('--match-iou-threshold', type=float, default=MATCH_IOU_THRESHOLD)

    parser.add_argument('--car-min-component-area', type=int, default=None)
    parser.add_argument('--car-min-instance-area', type=int, default=None)
    parser.add_argument('--car-min-seed-distance', type=int, default=None)
    parser.add_argument('--car-distance-peak-threshold', type=float, default=None)
    parser.add_argument('--car-distance-weight', type=float, default=None)
    parser.add_argument('--car-depth-edge-weight', type=float, default=None)
    parser.add_argument('--car-depth-marker-threshold', type=float, default=None)
    parser.add_argument('--car-min-depth-marker-area', type=int, default=None)
    parser.add_argument('--car-mean-instance-depth', type=float, default=None,
                        help='Drop car instance masks whose mean metric depth exceeds this value. Use <=0 to disable.')
    parser.add_argument('--car-max-instance-depth', type=float, dest='car_mean_instance_depth',
                        help=argparse.SUPPRESS)
    parser.add_argument('--no-car-shape-filter', action='store_true',
                        help='Disable post-watershed car instance shape filtering.')
    parser.add_argument('--car-min-aspect-ratio', type=float, default=None)
    parser.add_argument('--car-max-aspect-ratio', type=float, default=None)
    parser.add_argument('--car-min-rectangularity', type=float, default=None)
    parser.add_argument('--car-min-solidity', type=float, default=None)
    parser.add_argument('--human-min-component-area', type=int, default=None)
    parser.add_argument('--human-min-instance-area', type=int, default=None)
    parser.add_argument('--human-min-seed-distance', type=int, default=None)
    parser.add_argument('--human-distance-peak-threshold', type=float, default=None)
    parser.add_argument('--human-distance-weight', type=float, default=None)
    parser.add_argument('--human-depth-edge-weight', type=float, default=None)
    parser.add_argument('--human-depth-marker-threshold', type=float, default=None)
    parser.add_argument('--human-min-depth-marker-area', type=int, default=None)
    parser.add_argument('--human-mean-instance-depth', type=float, default=None,
                        help='Drop human instance masks whose mean metric depth exceeds this value. Use <=0 to disable.')
    parser.add_argument('--human-max-instance-depth', type=float, dest='human_mean_instance_depth',
                        help=argparse.SUPPRESS)
    parser.add_argument('--no-human-shape-filter', action='store_true',
                        help='Disable post-watershed human instance shape filtering.')
    parser.add_argument('--human-min-aspect-ratio', type=float, default=None)
    parser.add_argument('--human-max-aspect-ratio', type=float, default=None)
    parser.add_argument('--human-min-rectangularity', type=float, default=None)
    parser.add_argument('--human-min-solidity', type=float, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.eval_dataset:
        if args.data_root is None:
            raise ValueError('Please pass --data-root or set CITYSCAPES_ROOT.')
        run_dataset(args)
    else:
        run_single_image(args)
