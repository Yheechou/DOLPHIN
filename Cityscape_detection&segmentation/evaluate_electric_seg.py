import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import torch
import torchvision.transforms.functional as FF
from PIL import Image
from tqdm import tqdm

from utils.electric_seg import build_segmentation_model
from utils.cityscapes import Cityscapes


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Evaluation defaults. Edit these values directly when running from the IDE.
EXP_NAME = 'fcn_segmentation'
MODEL_CONFIG = {
    'model_name': 'fcn',  # 'fcn' or 'unet'
    'in_channels': 1,
    'num_classes': 8,
    'base_channels': 32,
}
USE_GRAYSCALE_INPUT = True
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'result_electric_seg', 'seg_fcn_gray', 'best.pth')
DATA_ROOT = "data/cityscapes"
SPLIT = 'train'
MAX_IMAGES = 500
OUTPUT_JSON = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SORT_DATASET = True
SAVE_VIS_NUM = 10
SAVE_VIS_DIR = None
CROP_TO_768 = True
CROP_SIZE = 768
OUTPUT_PATCH_SIZE = 16
SINGLE_IMAGE = None
SINGLE_TARGET = None

IGNORE_INDEX = 255
MODEL_CLASSES = ['sky', 'building', 'vegetation', 'road', 'car', 'human', 'traffic', 'sidewalk']
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


def get_result_dir(exp_name):
    return os.path.join(SCRIPT_DIR, 'result_electric_seg', exp_name)


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


def train_id_to_8class(target):
    mapped = np.full(target.shape, IGNORE_INDEX, dtype=np.int64)
    for class_index, class_name in enumerate(MODEL_CLASSES):
        for train_id in CITYSCAPES_CLASS_IDS[class_name]:
            mapped[target == train_id] = class_index
    return mapped


def target_to_color_image(target):
    color_image = np.zeros((*target.shape, 3), dtype=np.uint8)
    for class_index, class_name in enumerate(MODEL_CLASSES):
        color_image[target == class_index] = CLASS_COLORS[class_name]
    return color_image


def pred_to_masks(pred):
    return {
        class_name: (pred == class_index).astype(np.uint8)
        for class_index, class_name in enumerate(MODEL_CLASSES)
    }


def masks_to_color_image(seg_masks, model_class=MODEL_CLASSES):
    height, width = next(iter(seg_masks.values())).shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    for class_name in model_class:
        color_image[seg_masks[class_name].astype(bool)] = CLASS_COLORS[class_name]
    return color_image


def save_dataset_visualization(image, target, pred, image_path, output_dir, index):
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    prefix = f'{index + 1:06d}_{stem}'
    image.save(os.path.join(output_dir, f'{prefix}_image.png'))
    Image.fromarray(target_to_color_image(target)).save(os.path.join(output_dir, f'{prefix}_gt.png'))
    Image.fromarray(target_to_color_image(pred)).save(os.path.join(output_dir, f'{prefix}_pred.png'))


def degrade_prediction_to_patch_grid(pred, patch_size):
    if patch_size is None:
        return pred
    if patch_size <= 1:
        return pred

    height, width = pred.shape
    degraded = np.zeros_like(pred)
    num_classes = len(MODEL_CLASSES)
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = pred[y:y + patch_size, x:x + patch_size]
            counts = np.bincount(patch.reshape(-1), minlength=num_classes)
            degraded[y:y + patch_size, x:x + patch_size] = int(counts.argmax())
    return degraded


def predict(model, image, device, output_patch_size=None, use_grayscale_input=False):
    if use_grayscale_input:
        image = FF.rgb_to_grayscale(image, num_output_channels=1)
    image_tensor = FF.to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    pred = degrade_prediction_to_patch_grid(pred, output_patch_size)
    return pred


def get_checkpoint_model_config(checkpoint):
    config = dict(MODEL_CONFIG)
    args = checkpoint.get('args', {})
    for key in ('model_name', 'in_channels', 'num_classes', 'base_channels', 'use_grayscale_input'):
        if key in args:
            config[key] = args[key]
    if 'model_name' in checkpoint:
        config['model_name'] = checkpoint['model_name']
    return config


def load_model(checkpoint_path, device, model_config=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = get_checkpoint_model_config(checkpoint)
    if model_config:
        config.update({key: value for key, value in model_config.items() if value is not None})
    if config['num_classes'] != len(MODEL_CLASSES):
        raise ValueError(f'num_classes must be {len(MODEL_CLASSES)} for MODEL_CLASSES={MODEL_CLASSES}')

    model = build_segmentation_model(
        config['model_name'],
        in_channels=config['in_channels'],
        num_classes=config['num_classes'],
        base_channels=config['base_channels'],
    )
    state = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, config


def compute_iou_from_confusion(confusion):
    intersections = np.diag(confusion)
    unions = confusion.sum(axis=1) + confusion.sum(axis=0) - intersections
    ious = {}
    for class_index, class_name in enumerate(MODEL_CLASSES):
        ious[class_name] = float(intersections[class_index] / unions[class_index]) if unions[class_index] > 0 else float('nan')
    valid_ious = [iou for iou in ious.values() if not np.isnan(iou)]
    miou = float(np.mean(valid_ious)) if valid_ious else float('nan')
    return ious, miou


def compute_accuracy_from_confusion(confusion):
    correct = np.diag(confusion)
    total = confusion.sum()
    pa = float(correct.sum() / total) if total > 0 else float('nan')

    class_totals = confusion.sum(axis=1)
    class_accuracy = {}
    for class_index, class_name in enumerate(MODEL_CLASSES):
        class_accuracy[class_name] = (
            float(correct[class_index] / class_totals[class_index])
            if class_totals[class_index] > 0
            else float('nan')
        )
    valid_accuracy = [value for value in class_accuracy.values() if not np.isnan(value)]
    mpa = float(np.mean(valid_accuracy)) if valid_accuracy else float('nan')
    return pa, class_accuracy, mpa


def add_to_confusion(confusion, pred, target):
    valid = target != IGNORE_INDEX
    gt = target[valid].reshape(-1)
    pd = pred[valid].reshape(-1)
    if gt.size:
        encoded = len(MODEL_CLASSES) * gt + pd
        counts = np.bincount(encoded, minlength=len(MODEL_CLASSES) ** 2)
        confusion += counts.reshape(len(MODEL_CLASSES), len(MODEL_CLASSES))


def evaluate_single_image(image_path, target_path, model, device, result_dir,
                          crop_to_768=False, crop_size=CROP_SIZE, output_patch_size=None,
                          use_grayscale_input=False):
    image = Image.open(image_path).convert('RGB')
    if crop_to_768:
        image = center_crop_pil(image, crop_size)
    pred = predict(
        model,
        image,
        device,
        output_patch_size=output_patch_size,
        use_grayscale_input=use_grayscale_input,
    )

    os.makedirs(result_dir, exist_ok=True)
    image.save(os.path.join(result_dir, 'single_image.png'))
    Image.fromarray(target_to_color_image(pred)).save(os.path.join(result_dir, 'single_pred.png'))

    results = {
        'image_path': image_path,
        'target_path': target_path,
        'iou': None,
        'miou': None,
        'crop_to_768': crop_to_768,
        'crop_size': crop_size,
        'output_patch_size': output_patch_size,
        'use_grayscale_input': use_grayscale_input,
    }
    if target_path:
        target = Image.open(target_path)
        target = Cityscapes.encode_target(target)
        if crop_to_768:
            target = center_crop_array(target, crop_size)
        target = train_id_to_8class(target)
        confusion = np.zeros((len(MODEL_CLASSES), len(MODEL_CLASSES)), dtype=np.int64)
        add_to_confusion(confusion, pred, target)
        ious, miou = compute_iou_from_confusion(confusion)
        pa, class_accuracy, mpa = compute_accuracy_from_confusion(confusion)
        Image.fromarray(target_to_color_image(target)).save(os.path.join(result_dir, 'single_gt.png'))
        results.update({
            'iou': ious,
            'miou': miou,
            'pa': pa,
            'class_accuracy': class_accuracy,
            'mpa': mpa,
            'confusion_matrix': confusion.tolist(),
        })
    return results


def compute_dataset_iou(data_root, split, model, device, max_images=None, output_json=None,
                        save_vis_num=0, save_vis_dir=None, crop_to_768=False, crop_size=CROP_SIZE,
                        checkpoint_path=None, model_config=None, output_patch_size=None,
                        use_grayscale_input=False, sort_dataset=False):
    dataset = Cityscapes(data_root, split=split, target_type='semantic')
    if sort_dataset:
        dataset = sort_cityscapes_dataset(dataset)
    confusion = np.zeros((len(MODEL_CLASSES), len(MODEL_CLASSES)), dtype=np.int64)

    total = len(dataset) if max_images is None else min(max_images, len(dataset))
    progress = tqdm(range(total), desc='Evaluate electric seg', dynamic_ncols=True)
    for index in progress:
        image, target = dataset[index]
        image_rgb = image
        if crop_to_768:
            image_rgb = center_crop_pil(image_rgb, crop_size)
            target = center_crop_array(target, crop_size)

        target = train_id_to_8class(target)
        pred = predict(
            model,
            image_rgb,
            device,
            output_patch_size=output_patch_size,
            use_grayscale_input=use_grayscale_input,
        )
        add_to_confusion(confusion, pred, target)

        if save_vis_num and index < save_vis_num:
            save_dataset_visualization(
                image_rgb,
                target,
                pred,
                dataset.images[index],
                save_vis_dir,
                index,
            )
        progress.set_postfix(image=os.path.basename(dataset.images[index]))

    ious, miou = compute_iou_from_confusion(confusion)
    pa, class_accuracy, mpa = compute_accuracy_from_confusion(confusion)
    results = {
        'data_root': data_root,
        'split': split,
        'num_images': total,
        'checkpoint_path': checkpoint_path,
        'model_config': model_config,
        'iou': ious,
        'miou': miou,
        'pa': pa,
        'class_accuracy': class_accuracy,
        'mpa': mpa,
        'confusion_matrix': confusion.tolist(),
        'save_vis_num': save_vis_num,
        'save_vis_dir': save_vis_dir,
        'crop_to_768': crop_to_768,
        'crop_size': crop_size,
        'output_patch_size': output_patch_size,
        'use_grayscale_input': use_grayscale_input,
        'sort_dataset': sort_dataset,
    }

    print('\nIoU:')
    for class_name in MODEL_CLASSES:
        print(f'  {class_name}: {ious[class_name]:.6f}')
    print(f'mIoU: {miou:.6f}')
    print('\nPA:')
    print(f'  PA: {pa:.6f}')
    print('mPA:')
    for class_name in MODEL_CLASSES:
        print(f'  {class_name}: {class_accuracy[class_name]:.6f}')
    print(f'mPA: {mpa:.6f}')

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, 'w') as file:
            json.dump(results, file, indent=2)
        print(f'Saved metrics to {output_json}')

    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate FCN/U-Net 8-class electric segmentation on Cityscapes.')
    parser.add_argument('--exp-name', default=EXP_NAME,
                        help='Experiment name used as the result subdirectory.')
    parser.add_argument('--checkpoint-path', default=CHECKPOINT_PATH)
    parser.add_argument('--model-name', default=None, choices=['fcn', 'unet'],
                        help='Override model_name saved in the checkpoint.')
    parser.add_argument('--in-channels', type=int, default=None,
                        help='Override in_channels saved in the checkpoint.')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Override num_classes saved in the checkpoint.')
    parser.add_argument('--base-channels', type=int, default=None,
                        help='Override base_channels saved in the checkpoint.')
    parser.add_argument('--use-grayscale-input', action='store_true', default=USE_GRAYSCALE_INPUT,
                        help='Convert RGB images to grayscale before evaluation.')
    parser.add_argument('--use-rgb-input', action='store_false', dest='use_grayscale_input',
                        help='Use normal RGB images before evaluation.')
    parser.add_argument('--data-root', default=DATA_ROOT,
                        help='Cityscapes root containing leftImg8bit/ and gtFine/.')
    parser.add_argument('--split', default=SPLIT, choices=['train', 'val', 'test'])
    parser.add_argument('--max-images', type=int, default=MAX_IMAGES,
                        help='Evaluate only the first N images; useful for a quick test.')
    parser.add_argument('--output-json', default=OUTPUT_JSON)
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--sort-dataset', action='store_true', default=SORT_DATASET,
                        help='Sort dataset images by path before taking --max-images.')
    parser.add_argument('--no-sort-dataset', action='store_false', dest='sort_dataset',
                        help='Use the dataset order from utils.cityscapes.Cityscapes.')
    parser.add_argument('--save-vis-num', type=int, default=SAVE_VIS_NUM,
                        help='Save original image, GT and prediction visualizations for the first N images.')
    parser.add_argument('--save-vis-dir', default=SAVE_VIS_DIR,
                        help='Directory for dataset visualization images.')
    parser.add_argument('--crop-768', action='store_true', dest='crop_to_768', default=CROP_TO_768,
                        help='Evaluate the center 768x768 crop instead of the full image.')
    parser.add_argument('--no-crop-768', action='store_false', dest='crop_to_768',
                        help='Evaluate the full image.')
    parser.add_argument('--crop-size', type=int, default=CROP_SIZE)
    parser.add_argument('--output-patch-size', type=int, default=OUTPUT_PATCH_SIZE,
                        help='Optional patch-grid output size, e.g. 32/64/128. None keeps pixel-level predictions.')
    parser.add_argument('--single-image', default=SINGLE_IMAGE,
                        help='Optional image path for single-image visualization/evaluation mode.')
    parser.add_argument('--single-target', default=SINGLE_TARGET,
                        help='Optional Cityscapes labelIds target path for --single-image.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    result_dir = get_result_dir(args.exp_name)
    if args.output_json is None:
        args.output_json = os.path.join(result_dir, 'iou_metrics.json')
    if args.save_vis_dir is None:
        args.save_vis_dir = os.path.join(result_dir, 'visualizations')

    override_config = {
        'model_name': args.model_name,
        'in_channels': args.in_channels,
        'num_classes': args.num_classes,
        'base_channels': args.base_channels,
    }
    if args.use_grayscale_input and args.in_channels is None:
        override_config['in_channels'] = 1
    model, model_config = load_model(args.checkpoint_path, args.device, model_config=override_config)
    use_grayscale_input = args.use_grayscale_input or model_config.get('in_channels') == 1

    if args.single_image:
        single_results = evaluate_single_image(
            args.single_image,
            args.single_target,
            model,
            args.device,
            result_dir,
            crop_to_768=args.crop_to_768,
            crop_size=args.crop_size,
            output_patch_size=args.output_patch_size,
            use_grayscale_input=use_grayscale_input,
        )
        single_results['checkpoint_path'] = args.checkpoint_path
        single_results['model_config'] = model_config
        single_results['use_grayscale_input'] = use_grayscale_input
        with open(args.output_json, 'w') as file:
            json.dump(single_results, file, indent=2)
        print(f'Saved single-image result to {result_dir}')
        if single_results['miou'] is not None:
            print('\nIoU:')
            for class_name in MODEL_CLASSES:
                print(f'  {class_name}: {single_results["iou"][class_name]:.6f}')
            print(f'mIoU: {single_results["miou"]:.6f}')
            print('\nPA:')
            print(f'  PA: {single_results["pa"]:.6f}')
            print('mPA:')
            for class_name in MODEL_CLASSES:
                print(f'  {class_name}: {single_results["class_accuracy"][class_name]:.6f}')
            print(f'mPA: {single_results["mpa"]:.6f}')
    else:
        if args.data_root is None:
            raise ValueError('Please pass --data-root or set CITYSCAPES_ROOT.')
        compute_dataset_iou(
            args.data_root,
            args.split,
            model,
            args.device,
            max_images=args.max_images,
            output_json=args.output_json,
            save_vis_num=args.save_vis_num,
            save_vis_dir=args.save_vis_dir,
            crop_to_768=args.crop_to_768,
            crop_size=args.crop_size,
            checkpoint_path=args.checkpoint_path,
            model_config=model_config,
            output_patch_size=args.output_patch_size,
            use_grayscale_input=use_grayscale_input,
            sort_dataset=args.sort_dataset,
        )
