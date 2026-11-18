import argparse
import copy
import os


CONFIG = {
    "cuda_visible_devices": "3",
    "device": "cuda",
    "exp_name": "diffraction_evaluation",
    "root": "assets/airsim",
    "record_dir": "record",
    "test_dir": "Dataset/Test",
    "object_size": [256, 256],
    "distance_scale": 5,
    "batch_size": 1,
    "num_workers": 1,
    "load_ckpt_name": "ckpt_acc.pth",
    "conf_matrix_name": "conf_matrix.npy",
    "conf_matrix_img_name": "confusion_matrix_1.png",
    "model": {
        "prop_size": 400,
        "receive_size": 256,
        "coder_size": 256,
    },
}


def parse_size(value):
    parts = value.replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size should look like 100,100 or 100x100")
    return [int(parts[0]), int(parts[1])]


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--root", type=str)
    parser.add_argument("--cuda_visible_devices", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--object_size", type=parse_size)
    parser.add_argument("--distance_scale", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--load_ckpt_name", type=str)
    parser.add_argument("--conf_matrix_name", type=str)
    parser.add_argument("--conf_matrix_img_name", type=str)
    parser.add_argument("--prop_size", type=int)
    parser.add_argument("--receive_size", type=int)
    parser.add_argument("--coder_size", type=int)
    args, _ = parser.parse_known_args()

    config = copy.deepcopy(CONFIG)
    for key, value in vars(args).items():
        if value is None:
            continue
        if key in config["model"]:
            config["model"][key] = value
        else:
            config[key] = value
    return config


config = get_config()
os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.onn_experiment import Classification


def update_confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def build_testloader(config):
    object_size = config["object_size"]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((object_size[0], object_size[1])),
        transforms.ToTensor(),
    ])

    test_path = os.path.join(config["root"], config["test_dir"])
    test_dataset = datasets.ImageFolder(test_path, transform)
    testloader = DataLoader(
        test_dataset,
        num_workers=config["num_workers"],
        batch_size=config["batch_size"],
        shuffle=False,
    )
    return testloader, test_dataset.classes


def compute_precision_recall(conf_matrix):
    precision = []
    recall = []
    for i in range(len(conf_matrix)):
        tp = conf_matrix[i, i]
        fp = sum(conf_matrix[i, :]) - tp
        fn = sum(conf_matrix[:, i]) - tp
        precision.append(tp / (tp + fp) if tp + fp > 0 else 0)
        recall.append(tp / (tp + fn) if tp + fn > 0 else 0)
    return precision, recall


def save_confusion_matrix(conf_matrix, class_names, output_path):
    original_cmap = plt.get_cmap("OrRd")
    lower_half = original_cmap(np.linspace(0.02, 0.25, 256))
    new_cmap = LinearSegmentedColormap.from_list("RdPu_half", lower_half)

    plt.figure(figsize=(12, 6))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=new_cmap)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.axis("off")

    font_path = "./Arial.ttf"
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path, size=25, weight="bold")
    else:
        font_prop = fm.FontProperties(size=25, weight="bold")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(
                j,
                i,
                str(int(conf_matrix[i, j])),
                horizontalalignment="center",
                verticalalignment="center",
                fontproperties=font_prop,
            )

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    record_path = os.path.join(config["root"], config["record_dir"], config["exp_name"])
    testloader, class_names = build_testloader(config)

    model = Classification(
        object_size=config["object_size"],
        device=str(device),
        distance_scale=config["distance_scale"],
        model_config=config["model"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    conf_matrix = torch.zeros(len(class_names), len(class_names))

    checkpoint = torch.load(os.path.join(record_path, config["load_ckpt_name"]), map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    running_test_correct = 0
    running_test_total = 0
    running_test_loss = 0.0

    print(config["exp_name"])
    with torch.no_grad():
        with tqdm(enumerate(testloader, 0), total=len(testloader), desc="Evaluating", leave=False) as test_pbar:
            for i, (objects, labels) in test_pbar:
                objects, labels = objects.to(device), labels.to(device)
                outputs = model(objects)
                update_confusion_matrix(outputs, labels, conf_matrix)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                running_test_loss += loss.item()
                running_test_total += labels.size(0)
                running_test_correct += (predicted == labels).sum().item()

                test_pbar.set_postfix({
                    "Test Accuracy": f"{100 * running_test_correct / running_test_total:.2f}%",
                    "Test Loss": f"{running_test_loss / (i + 1):.4f}",
                })

    accuracy = 100 * running_test_correct / running_test_total
    average_loss = running_test_loss / len(testloader)
    print(f"Accuracy: {accuracy:.2f}%, Average Loss: {average_loss:.4f}")
    print(conf_matrix)

    conf_matrix = conf_matrix.cpu().numpy()
    precision, recall = compute_precision_recall(conf_matrix)
    for class_name, precision_value, recall_value in zip(class_names, precision, recall):
        print(f"{class_name} 的精确率为: {precision_value:.2f}, 召回率为: {recall_value:.2f}")

    conf_matrix_path = os.path.join(record_path, config["conf_matrix_name"])
    np.save(conf_matrix_path, conf_matrix)
    save_confusion_matrix(
        conf_matrix,
        [str(i) for i in range(len(conf_matrix))],
        os.path.join(record_path, config["conf_matrix_img_name"]),
    )


if __name__ == "__main__":
    main()
