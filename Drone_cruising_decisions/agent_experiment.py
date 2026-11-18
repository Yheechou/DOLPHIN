import argparse
import copy
import os


CONFIG = {
    "cuda_visible_devices": "2",
    "device": "cuda",
    "exp_name": "diffraction_training",
    "root": "assets/airsim",
    "record_dir": "record",
    "train_dir": "Dataset/Train",
    "test_dir": "Dataset/Test",
    "object_size": [256, 256],
    "distance_scale": 40,
    "batch_size": 1,
    "num_workers": 1,
    "num_epochs": 100,
    "lr": 0.001,
    "load_ckpt": False,
    "load_ckpt_name": "ckpt.pth",
    "save_ckpt_name": "ckpt_acc.pth",
    "model": {
        "prop_size": 400,
        "receive_size": 256,
        "coder_size": 256,
    },
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("bool value expected")


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
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--load_ckpt", type=str2bool)
    parser.add_argument("--load_ckpt_name", type=str)
    parser.add_argument("--save_ckpt_name", type=str)
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

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.onn_experiment import Classification


def build_dataloaders(config):
    object_size = config["object_size"]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((object_size[0], object_size[1])),
        transforms.ToTensor(),
    ])

    train_path = os.path.join(config["root"], config["train_dir"])
    test_path = os.path.join(config["root"], config["test_dir"])
    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    trainloader = DataLoader(
        train_dataset,
        num_workers=config["num_workers"],
        batch_size=config["batch_size"],
        shuffle=True,
    )
    testloader = DataLoader(
        test_dataset,
        num_workers=config["num_workers"],
        batch_size=config["batch_size"],
        shuffle=False,
    )
    return trainloader, testloader


def write_config(file_path, config):
    with open(file_path, "w", encoding="utf-8") as file:
        for key, value in config.items():
            file.write(f"{key}:{value}\n")


def evaluate(model, testloader, criterion, device):
    model.eval()
    running_test_correct = 0
    running_test_total = 0
    running_test_loss = 0.0

    with torch.no_grad():
        with tqdm(enumerate(testloader, 0), total=len(testloader), desc="Evaluating", leave=False) as test_pbar:
            for i, (objects, labels) in test_pbar:
                objects, labels = objects.to(device), labels.to(device)
                outputs = model(objects)
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
    return accuracy, average_loss


def main():
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    record_path = os.path.join(config["root"], config["record_dir"], config["exp_name"])
    os.makedirs(record_path, exist_ok=True)

    print(config["exp_name"])
    file_path = os.path.join(record_path, "record.txt")
    write_config(file_path, config)

    trainloader, testloader = build_dataloaders(config)
    model = Classification(
        object_size=config["object_size"],
        device=str(device),
        distance_scale=config["distance_scale"],
        model_config=config["model"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    if config["load_ckpt"]:
        checkpoint = torch.load(os.path.join(record_path, config["load_ckpt_name"]), map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    best_accuracy = 0.0

    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        with tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f"Epoch [{epoch+1}/{config['num_epochs']}]") as pbar:
            for i, (objects, labels) in pbar:
                optimizer.zero_grad()
                objects, labels = objects.to(device), labels.to(device)
                outputs = model(objects)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                running_total += labels.size(0)
                running_correct += (predicted == labels).sum().item()

                pbar.set_postfix({
                    "Current Loss": f"{loss.item():.4f}",
                    "Running Loss": f"{running_loss / (i + 1):.4f}",
                    "Running Accuracy": f"{100 * running_correct / running_total:.2f}%",
                })

        with open(file_path, "a", encoding="utf-8") as file:
            file.write(
                f"Train epoch [{epoch+1}/{config['num_epochs']}], "
                f"Average Loss: {running_loss / (i + 1):.4f}, "
                f"Accuracy: {100 * running_correct / running_total:.2f}%\n"
            )

        accuracy, average_loss = evaluate(model, testloader, criterion, device)
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Accuracy: {accuracy:.2f}%, Average Loss: {average_loss:.4f}")
        with open(file_path, "a", encoding="utf-8") as file:
            file.write(
                f"Test Epoch [{epoch+1}/{config['num_epochs']}], "
                f"Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%\n"
            )

        if 100 * running_correct / running_total > best_accuracy:
            best_accuracy = 100 * running_correct / running_total
            state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            torch.save(state, os.path.join(record_path, config["save_ckpt_name"]))


if __name__ == "__main__":
    main()
