import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as FF
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from utils.cityscapes import Cityscapes


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Edit these defaults directly when running from the IDE.
EXP_NAME = 'pretrained_detection'
BACKEND = 'fasterrcnn'  # 'fasterrcnn' or 'yolo'
MODEL_WEIGHTS = None  # Faster R-CNN uses torchvision default if None. YOLO can use 'yolo11s.pt', 'yolov8s.pt', etc.
EVAL_DATASET = True
DATA_ROOT = "data/cityscapes"
SPLIT = 'train'
MAX_IMAGES = 500
OUTPUT_JSON = None
SAVE_VIS_NUM = 10
SORT_DATASET = True
IMAGE_PATH = 'data/cityscapes/leftImg8bit/train/aachen/aachen_000171_000019_leftImg8bit.png'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CROP_TO_768 = True
CROP_SIZE = 768
CONF_THRESHOLD = 0.5
MATCH_IOU_THRESHOLD = 0.5
BOX_THICKNESS = 3
PRED_MIN_AREA = {
    'car': 0,
    'human': 0,
}
GT_MIN_AREA = {
    'car': 5000,
    'human': 2000,
}
DETECTION_CLASSES = ['car', 'human']
CLASS_COLORS = {
    'car': (142, 0, 0),
    'human': (60, 20, 220),
}
CITYSCAPES_CLASS_IDS = {
    'car': (26,),
    'human': (24, 25),
}
TORCHVISION_COCO_TO_TARGET = {
    1: 'human',  # person
    3: 'car',
}
YOLO_COCO_TO_TARGET = {
    'person': 'human',
    'car': 'car',
}


def get_result_dir(exp_name):
    return os.path.join(SCRIPT_DIR, 'result_pretrained_detect', exp_name)


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


def get_instance_path_from_image(image_path):
    return image_path.replace('/leftImg8bit/', '/gtFine/').replace('_leftImg8bit.png', '_gtFine_instanceIds.png')


def get_label_path_from_image(image_path):
    return image_path.replace('/leftImg8bit/', '/gtFine/').replace('_leftImg8bit.png', '_gtFine_labelIds.png')


def get_gt_min_area(args):
    gt_min_area = GT_MIN_AREA.copy()
    if args.car_gt_min_area is not None:
        gt_min_area['car'] = args.car_gt_min_area
    if args.human_gt_min_area is not None:
        gt_min_area['human'] = args.human_gt_min_area
    return gt_min_area


def get_pred_min_area(args):
    pred_min_area = PRED_MIN_AREA.copy()
    if args.car_pred_min_area is not None:
        pred_min_area['car'] = args.car_pred_min_area
    if args.human_pred_min_area is not None:
        pred_min_area['human'] = args.human_pred_min_area
    return pred_min_area


def bbox_area_xyxy(box):
    x1, y1, x2, y2 = box
    return max(0.0, float(x2) - float(x1) + 1.0) * max(0.0, float(y2) - float(y1) + 1.0)


def filter_detections_by_area(detections, pred_min_area):
    filtered = []
    for det in detections:
        area = bbox_area_xyxy(det['bbox_xyxy'])
        if area < pred_min_area.get(det['class'], 0):
            continue
        det = det.copy()
        det['area'] = area
        filtered.append(det)
    return filtered


def load_gt_instance(image_path, crop_to_768=True, crop_size=768):
    instance_path = get_instance_path_from_image(image_path)
    label_path = get_label_path_from_image(image_path)
    if os.path.exists(instance_path):
        gt = np.array(Image.open(instance_path), dtype=np.int64)
        use_instance_ids = True
    elif os.path.exists(label_path):
        gt = np.array(Image.open(label_path), dtype=np.int64)
        use_instance_ids = False
    else:
        return None, False

    if crop_to_768:
        gt = center_crop_array(gt, crop_size)
    return gt, use_instance_ids


def load_gt_boxes(image_path, crop_to_768=True, crop_size=768, gt_min_area=None):
    gt, use_instance_ids = load_gt_instance(image_path, crop_to_768=crop_to_768, crop_size=crop_size)
    if gt is None:
        return None
    if gt_min_area is None:
        gt_min_area = GT_MIN_AREA

    boxes = {class_name: [] for class_name in DETECTION_CLASSES}
    for class_name, class_ids in CITYSCAPES_CLASS_IDS.items():
        min_area = gt_min_area.get(class_name, 0)
        if use_instance_ids:
            instance_ids = np.unique(gt[np.isin(gt // 1000, class_ids)])
            instance_ids = [value for value in instance_ids if value >= 1000]
            for instance_id in instance_ids:
                mask = gt == instance_id
                ys, xs = np.where(mask)
                area = int(mask.sum())
                if area < min_area or area == 0:
                    continue
                boxes[class_name].append([
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()),
                    int(ys.max()),
                ])
            continue
        else:
            mask = np.isin(gt, class_ids)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            boxes[class_name].append([x, y, x + width - 1, y + height - 1])
    return boxes


def gt_instance_to_color(gt, use_instance_ids):
    color_image = np.zeros((*gt.shape, 3), dtype=np.uint8)
    color_index = 0
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 128, 0), (0, 128, 128),
        (255, 128, 0), (128, 0, 255), (0, 128, 255), (255, 0, 128),
    ]
    for class_name, class_ids in CITYSCAPES_CLASS_IDS.items():
        if use_instance_ids:
            instance_ids = np.unique(gt[np.isin(gt // 1000, class_ids)])
            instance_ids = [value for value in instance_ids if value >= 1000]
            for instance_id in instance_ids:
                color_image[gt == instance_id] = palette[color_index % len(palette)]
                color_index += 1
        else:
            mask = np.isin(gt, class_ids).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
            for label in range(1, num_labels):
                color_image[labels == label] = palette[color_index % len(palette)]
                color_index += 1
    return color_image


def boxes_to_detections(gt_boxes):
    detections = []
    for class_name, boxes in gt_boxes.items():
        for box in boxes:
            detections.append({
                'class': class_name,
                'bbox_xyxy': box,
                'score': 1.0,
            })
    return detections


def draw_detection_boxes(image, detections, thickness=3, show_score=True):
    drawn = image.copy()
    draw = ImageDraw.Draw(drawn)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 18)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        class_name = det['class']
        color = CLASS_COLORS[class_name]
        x1, y1, x2, y2 = [int(round(v)) for v in det['bbox_xyxy']]
        for offset in range(thickness):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)
        label = class_name
        if show_score and 'score' in det:
            label = f'{class_name} {det["score"]:.2f}'
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_h = text_box[3] - text_box[1]
        text_w = text_box[2] - text_box[0]
        bg_y1 = max(0, y1 - text_h - 4)
        draw.rectangle([x1, bg_y1, x1 + text_w + 4, bg_y1 + text_h + 4], fill=color)
        draw.text((x1 + 2, bg_y1 + 2), label, fill=(255, 255, 255), font=font)
    return drawn


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def compute_class_counts(predictions, gt_by_image, class_name, iou_threshold=0.5):
    class_predictions = [pred for pred in predictions if pred['class'] == class_name]
    class_predictions = sorted(class_predictions, key=lambda pred: pred.get('score', 0.0), reverse=True)
    matched = {
        image_id: np.zeros(len(gt_boxes.get(class_name, [])), dtype=bool)
        for image_id, gt_boxes in gt_by_image.items()
    }
    true_positive = 0
    false_positive = 0

    for pred in class_predictions:
        image_id = pred['image_id']
        gt_boxes = gt_by_image.get(image_id, {}).get(class_name, [])
        best_iou = 0.0
        best_gt_index = -1
        for gt_index, gt_box in enumerate(gt_boxes):
            if matched[image_id][gt_index]:
                continue
            iou = box_iou(pred['bbox_xyxy'], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_iou >= iou_threshold and best_gt_index >= 0:
            true_positive += 1
            matched[image_id][best_gt_index] = True
        else:
            false_positive += 1

    num_gt = int(sum(len(gt_boxes.get(class_name, [])) for gt_boxes in gt_by_image.values()))
    false_negative = num_gt - true_positive
    return true_positive, false_positive, false_negative, num_gt


def compute_class_tp_center_distances(predictions, gt_by_image, class_name, iou_threshold=0.5):
    distances = []
    class_predictions = [pred for pred in predictions if pred['class'] == class_name]
    class_predictions = sorted(class_predictions, key=lambda pred: pred.get('score', 0.0), reverse=True)
    matched = {
        image_id: np.zeros(len(gt_boxes.get(class_name, [])), dtype=bool)
        for image_id, gt_boxes in gt_by_image.items()
    }

    for pred in class_predictions:
        image_id = pred['image_id']
        gt_boxes = gt_by_image.get(image_id, {}).get(class_name, [])
        best_iou = 0.0
        best_gt_index = -1
        for gt_index, gt_box in enumerate(gt_boxes):
            if matched[image_id][gt_index]:
                continue
            iou = box_iou(pred['bbox_xyxy'], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_iou >= iou_threshold and best_gt_index >= 0:
            matched[image_id][best_gt_index] = True
            pred_center = box_center(pred['bbox_xyxy'])
            gt_center = box_center(gt_boxes[best_gt_index])
            distances.append(float(np.linalg.norm(pred_center - gt_center)))

    avg_distance = float(np.mean(distances)) if distances else float('nan')
    return avg_distance, distances


def compute_class_nearest_gt_center_distances(predictions, gt_by_image, class_name):
    """
    For every predicted box of this class, find the nearest GT box center
    from the same image and same class, then compute the Euclidean distance.

    Unlike compute_class_tp_center_distances(), this metric does not require
    IoU matching and does not do one-to-one matching, so FP predictions are
    also included as long as that image has at least one GT box of the same class.
    """
    distances = []
    class_predictions = [pred for pred in predictions if pred['class'] == class_name]

    for pred in class_predictions:
        image_id = pred['image_id']
        gt_boxes = gt_by_image.get(image_id, {}).get(class_name, [])
        if not gt_boxes:
            continue

        pred_center = box_center(pred['bbox_xyxy'])
        gt_centers = np.stack([box_center(gt_box) for gt_box in gt_boxes], axis=0)
        dists = np.linalg.norm(gt_centers - pred_center[None, :], axis=1)
        distances.append(float(dists.min()))

    avg_distance = float(np.mean(distances)) if distances else float('nan')
    return avg_distance, distances


def counts_to_scores(true_positive, false_positive, false_negative):
    precision_den = true_positive + false_positive
    recall_den = true_positive + false_negative
    precision = true_positive / precision_den if precision_den > 0 else float('nan')
    recall = true_positive / recall_den if recall_den > 0 else float('nan')
    f1_den = precision + recall
    f1 = 2 * precision * recall / f1_den if f1_den > 0 and not np.isnan(f1_den) else float('nan')
    return precision, recall, f1


def compute_detection_metrics(predictions, gt_by_image, iou_threshold):
    per_class = {}
    totals = {'tp': 0, 'fp': 0, 'fn': 0, 'num_gt': 0}
    all_tp_center_distances = []
    all_nearest_gt_center_distances = []
    for class_name in DETECTION_CLASSES:
        tp, fp, fn, num_gt = compute_class_counts(predictions, gt_by_image, class_name, iou_threshold=iou_threshold)
        precision, recall, f1 = counts_to_scores(tp, fp, fn)
        avg_tp_center_distance, tp_center_distances = compute_class_tp_center_distances(
            predictions,
            gt_by_image,
            class_name,
            iou_threshold=iou_threshold,
        )
        avg_nearest_gt_center_distance, nearest_gt_center_distances = compute_class_nearest_gt_center_distances(
            predictions,
            gt_by_image,
            class_name,
        )
        per_class[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'num_gt': num_gt,
            'avg_tp_center_distance': avg_tp_center_distance,
            'tp_center_distance_count': len(tp_center_distances),
            'avg_nearest_gt_center_distance': avg_nearest_gt_center_distance,
            'nearest_gt_center_distance_count': len(nearest_gt_center_distances),
        }
        totals['tp'] += tp
        totals['fp'] += fp
        totals['fn'] += fn
        totals['num_gt'] += num_gt
        all_tp_center_distances.extend(tp_center_distances)
        all_nearest_gt_center_distances.extend(nearest_gt_center_distances)

    precision, recall, f1 = counts_to_scores(totals['tp'], totals['fp'], totals['fn'])
    overall = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_tp_center_distance': (
            float(np.mean(all_tp_center_distances)) if all_tp_center_distances else float('nan')
        ),
        'tp_center_distance_count': len(all_tp_center_distances),
        'avg_nearest_gt_center_distance': (
            float(np.mean(all_nearest_gt_center_distances)) if all_nearest_gt_center_distances else float('nan')
        ),
        'nearest_gt_center_distance_count': len(all_nearest_gt_center_distances),
        **totals,
    }
    num_gt = {class_name: values['num_gt'] for class_name, values in per_class.items()}
    return per_class, overall, num_gt


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def load_fasterrcnn(args):
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if args.model_weights is None else args.model_weights
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    model.to(args.device)
    model.eval()
    return model


def load_yolo(args):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            'Ultralytics is not installed in this environment. Install it or use BACKEND="fasterrcnn".'
        ) from exc

    weights = args.model_weights or 'yolo11s.pt'
    model = YOLO(weights)
    return model


def load_detector(args):
    if args.backend == 'fasterrcnn':
        return load_fasterrcnn(args)
    if args.backend == 'yolo':
        return load_yolo(args)
    raise ValueError(f'Unknown backend "{args.backend}"')


def predict_fasterrcnn(model, image, args):
    image_tensor = FF.to_tensor(image).to(args.device)
    with torch.no_grad():
        output = model([image_tensor])[0]

    detections = []
    for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
        score = float(score.item())
        label = int(label.item())
        if score < args.conf_threshold or label not in TORCHVISION_COCO_TO_TARGET:
            continue
        x1, y1, x2, y2 = box.detach().cpu().numpy().tolist()
        detections.append({
            'class': TORCHVISION_COCO_TO_TARGET[label],
            'bbox_xyxy': [float(x1), float(y1), float(x2), float(y2)],
            'score': score,
            'source_class': label,
        })
    return detections


def predict_yolo(model, image, args):
    results = model.predict(
        source=np.array(image),
        conf=args.conf_threshold,
        device=args.device,
        verbose=False,
    )
    names = model.names
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        labels = result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, label in zip(boxes, scores, labels):
            source_name = names[int(label)]
            if source_name not in YOLO_COCO_TO_TARGET:
                continue
            detections.append({
                'class': YOLO_COCO_TO_TARGET[source_name],
                'bbox_xyxy': [float(v) for v in box.tolist()],
                'score': float(score),
                'source_class': source_name,
            })
    return detections


def predict_detections(model, image, args):
    if args.backend == 'fasterrcnn':
        return predict_fasterrcnn(model, image, args)
    if args.backend == 'yolo':
        return predict_yolo(model, image, args)
    raise ValueError(f'Unknown backend "{args.backend}"')


def load_image(image_path, crop_to_768=True, crop_size=768):
    image = Image.open(image_path).convert('RGB')
    if crop_to_768:
        image = center_crop_pil(image, crop_size)
    return image


def load_gt_visualizations(image_path, image, crop_to_768=True, crop_size=768, gt_min_area=None, thickness=3):
    gt, use_instance_ids = load_gt_instance(image_path, crop_to_768=crop_to_768, crop_size=crop_size)
    if gt is None:
        return None, None, None
    gt_instance = gt_instance_to_color(gt, use_instance_ids)
    gt_boxes = load_gt_boxes(image_path, crop_to_768=crop_to_768, crop_size=crop_size, gt_min_area=gt_min_area)
    gt_detection = draw_detection_boxes(image, boxes_to_detections(gt_boxes), thickness=thickness, show_score=False)
    return gt_instance, gt_detection, gt_boxes


def save_visualizations(result_dir, image_id, image, detections, gt_instance=None, gt_detection=None,
                        thickness=3):
    vis_dir = os.path.join(result_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    image.save(os.path.join(vis_dir, f'{image_id}_image.png'))
    draw_detection_boxes(image, detections, thickness=thickness).save(os.path.join(vis_dir, f'{image_id}_pred_detection.png'))
    if gt_instance is not None:
        Image.fromarray(gt_instance).save(os.path.join(vis_dir, f'{image_id}_gt_instance.png'))
    if gt_detection is not None:
        gt_detection.save(os.path.join(vis_dir, f'{image_id}_gt_detection.png'))


def run_single_image(args):
    result_dir = get_result_dir(args.exp_name)
    os.makedirs(result_dir, exist_ok=True)
    if args.output_json is None:
        args.output_json = os.path.join(result_dir, 'detection_results.json')

    detector = load_detector(args)
    gt_min_area = get_gt_min_area(args)
    pred_min_area = get_pred_min_area(args)
    image = load_image(args.image_path, crop_to_768=args.crop_to_768, crop_size=args.crop_size)
    detections = filter_detections_by_area(predict_detections(detector, image, args), pred_min_area)
    image_id = os.path.splitext(os.path.basename(args.image_path))[0]

    gt_instance, gt_detection, gt_boxes = load_gt_visualizations(
        args.image_path,
        image,
        crop_to_768=args.crop_to_768,
        crop_size=args.crop_size,
        gt_min_area=gt_min_area,
        thickness=args.box_thickness,
    )
    save_visualizations(
        result_dir,
        image_id,
        image,
        detections,
        gt_instance=gt_instance,
        gt_detection=gt_detection,
        thickness=args.box_thickness,
    )

    metric_predictions = []
    gt_by_image = {}
    has_gt = gt_boxes is not None
    metrics_per_class = {
        class_name: {
            'precision': float('nan'),
            'recall': float('nan'),
            'f1': float('nan'),
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'num_gt': 0,
            'avg_tp_center_distance': float('nan'),
            'tp_center_distance_count': 0,
            'avg_nearest_gt_center_distance': float('nan'),
            'nearest_gt_center_distance_count': 0,
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
        'avg_tp_center_distance': float('nan'),
        'tp_center_distance_count': 0,
        'avg_nearest_gt_center_distance': float('nan'),
        'nearest_gt_center_distance_count': 0,
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
            'backend': args.backend,
            'model_weights': args.model_weights,
            'conf_threshold': args.conf_threshold,
            'match_iou_threshold': args.match_iou_threshold,
            'pred_min_area': pred_min_area,
            'gt_min_area': gt_min_area,
            'crop_to_768': args.crop_to_768,
            'crop_size': args.crop_size,
        },
        'metrics': {
            'overall': overall_metrics,
            'per_class': metrics_per_class,
            'precision': overall_metrics['precision'],
            'recall': overall_metrics['recall'],
            'f1': overall_metrics['f1'],
            'avg_tp_center_distance': overall_metrics['avg_tp_center_distance'],
            'avg_nearest_gt_center_distance': overall_metrics['avg_nearest_gt_center_distance'],
            'num_gt': num_gt,
        },
        'image': {
            'image_path': args.image_path,
            'has_gt': has_gt,
        },
        'detections': detections,
        'gt_boxes': gt_boxes,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w') as file:
        json.dump(to_jsonable(results), file, indent=2)

    print(f'Saved visualizations to {os.path.join(result_dir, "visualizations")}')
    print(f'Saved detections to {args.output_json}')
    print(f"Overall precision/recall/F1@{args.match_iou_threshold}: "
          f"{overall_metrics['precision']:.6f}/"
          f"{overall_metrics['recall']:.6f}/"
          f"{overall_metrics['f1']:.6f}")
    print(f"Average TP center distance: {overall_metrics['avg_tp_center_distance']:.6f}")
    print(f"Average nearest-GT center distance for all predictions: "
          f"{overall_metrics['avg_nearest_gt_center_distance']:.6f}")


def run_dataset(args):
    if args.data_root is None:
        raise ValueError('Please pass --data-root or set CITYSCAPES_ROOT.')

    result_dir = get_result_dir(args.exp_name)
    os.makedirs(result_dir, exist_ok=True)
    if args.output_json is None:
        args.output_json = os.path.join(result_dir, 'precision_recall_f1_metrics.json')

    detector = load_detector(args)
    dataset = Cityscapes(args.data_root, split=args.split, target_type='semantic')
    if args.sort_dataset:
        dataset = sort_cityscapes_dataset(dataset)
    total = len(dataset) if args.max_images is None else min(args.max_images, len(dataset))
    gt_min_area = get_gt_min_area(args)
    pred_min_area = get_pred_min_area(args)
    predictions = []
    gt_by_image = {}

    progress = tqdm(range(total), desc=f'Evaluate {args.backend}', dynamic_ncols=True)
    for index in progress:
        image_path = dataset.images[index]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        image = load_image(image_path, crop_to_768=args.crop_to_768, crop_size=args.crop_size)
        detections = filter_detections_by_area(predict_detections(detector, image, args), pred_min_area)

        for det in detections:
            det = det.copy()
            det['image_id'] = image_id
            predictions.append(det)

        gt_instance, gt_detection, gt_boxes = load_gt_visualizations(
            image_path,
            image,
            crop_to_768=args.crop_to_768,
            crop_size=args.crop_size,
            gt_min_area=gt_min_area,
            thickness=args.box_thickness,
        )
        if gt_boxes is not None:
            gt_by_image[image_id] = gt_boxes

        if args.save_vis_num and index < args.save_vis_num:
            save_visualizations(
                result_dir,
                image_id,
                image,
                detections,
                gt_instance=gt_instance,
                gt_detection=gt_detection,
                thickness=args.box_thickness,
            )

        progress.set_postfix(image=os.path.basename(image_path), detections=len(detections))

    metrics_per_class, overall_metrics, num_gt = compute_detection_metrics(
        predictions,
        gt_by_image,
        args.match_iou_threshold,
    )
    metrics = {
        'parameters': {
            'exp_name': args.exp_name,
            'mode': 'dataset',
            'backend': args.backend,
            'model_weights': args.model_weights,
            'data_root': args.data_root,
            'split': args.split,
            'max_images': args.max_images,
            'sort_dataset': args.sort_dataset,
            'conf_threshold': args.conf_threshold,
            'match_iou_threshold': args.match_iou_threshold,
            'pred_min_area': pred_min_area,
            'gt_min_area': gt_min_area,
            'crop_to_768': args.crop_to_768,
            'crop_size': args.crop_size,
        },
        'metrics': {
            'overall': overall_metrics,
            'per_class': metrics_per_class,
            'precision': overall_metrics['precision'],
            'recall': overall_metrics['recall'],
            'f1': overall_metrics['f1'],
            'avg_tp_center_distance': overall_metrics['avg_tp_center_distance'],
            'avg_nearest_gt_center_distance': overall_metrics['avg_nearest_gt_center_distance'],
            'num_gt': num_gt,
            'num_predictions': len(predictions),
        },
        'dataset': {
            'num_images': total,
            'num_images_with_gt': len(gt_by_image),
        },
        'detections': predictions,
        'gt_boxes': gt_by_image,
    }

    print('\nDetection Metrics:')
    for class_name, values in metrics_per_class.items():
        print(
            f"  {class_name}: precision={values['precision']:.6f}, "
            f"recall={values['recall']:.6f}, f1={values['f1']:.6f}, "
            f"avg_tp_center_distance={values['avg_tp_center_distance']:.6f}, "
            f"avg_nearest_gt_center_distance={values['avg_nearest_gt_center_distance']:.6f}, "
            f"tp={values['tp']}, fp={values['fp']}, fn={values['fn']}"
        )
    print(
        f"Overall@{args.match_iou_threshold}: "
        f"precision={overall_metrics['precision']:.6f}, "
        f"recall={overall_metrics['recall']:.6f}, "
        f"f1={overall_metrics['f1']:.6f}, "
        f"avg_tp_center_distance={overall_metrics['avg_tp_center_distance']:.6f}, "
        f"avg_nearest_gt_center_distance={overall_metrics['avg_nearest_gt_center_distance']:.6f}"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w') as file:
        json.dump(to_jsonable(metrics), file, indent=2)
    print(f'Saved metrics to {args.output_json}')


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate pretrained YOLO/Faster R-CNN detection baseline on Cityscapes.')
    parser.add_argument('--exp-name', default=EXP_NAME)
    parser.add_argument('--backend', default=BACKEND, choices=['fasterrcnn', 'yolo'])
    parser.add_argument('--model-weights', default=MODEL_WEIGHTS,
                        help='Faster R-CNN: leave None for torchvision default. YOLO: e.g. yolo11s.pt.')
    parser.add_argument('--eval-dataset', action='store_true', default=EVAL_DATASET,
                        help='Evaluate precision/recall/F1 on a Cityscapes split instead of one image.')
    parser.add_argument('--single-image', action='store_false', dest='eval_dataset',
                        help='Run one image even if EVAL_DATASET is True in the file defaults.')
    parser.add_argument('--data-root', default=DATA_ROOT,
                        help='Cityscapes root containing leftImg8bit/ and gtFine/.')
    parser.add_argument('--split', default=SPLIT, choices=['train', 'val', 'test'])
    parser.add_argument('--max-images', type=int, default=MAX_IMAGES)
    parser.add_argument('--output-json', default=OUTPUT_JSON)
    parser.add_argument('--save-vis-num', type=int, default=SAVE_VIS_NUM)
    parser.add_argument('--sort-dataset', action='store_true', default=SORT_DATASET,
                        help='Sort dataset images by path before taking --max-images.')
    parser.add_argument('--no-sort-dataset', action='store_false', dest='sort_dataset',
                        help='Use the dataset order from utils.cityscapes.Cityscapes.')
    parser.add_argument('--image-path', default=IMAGE_PATH)
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--crop-768', action='store_true', dest='crop_to_768', default=CROP_TO_768)
    parser.add_argument('--no-crop-768', action='store_false', dest='crop_to_768')
    parser.add_argument('--crop-size', type=int, default=CROP_SIZE)
    parser.add_argument('--conf-threshold', type=float, default=CONF_THRESHOLD)
    parser.add_argument('--match-iou-threshold', type=float, default=MATCH_IOU_THRESHOLD)
    parser.add_argument('--box-thickness', type=int, default=BOX_THICKNESS)
    parser.add_argument('--car-pred-min-area', type=int, default=None,
                        help='Drop predicted car boxes smaller than this pixel area.')
    parser.add_argument('--human-pred-min-area', type=int, default=None,
                        help='Drop predicted human boxes smaller than this pixel area.')
    parser.add_argument('--car-gt-min-area', type=int, default=None)
    parser.add_argument('--human-gt-min-area', type=int, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.eval_dataset:
        run_dataset(args)
    else:
        run_single_image(args)
