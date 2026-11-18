import argparse
import csv
import gzip
import os
import struct

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.model_diff_forward import ONN_diff
from utils.model_multi_layer_forward import ONN_multi


class PCAMNISTDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def read_idx_images(path):
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num_images, rows * cols).astype(np.float32) / 255.0


def read_idx_labels(path):
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file: {path}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)


def load_mnist_raw(root):
    raw_dir = os.path.join(root, "MNIST", "raw")
    train_images = read_idx_images(os.path.join(raw_dir, "train-images-idx3-ubyte"))
    train_labels = read_idx_labels(os.path.join(raw_dir, "train-labels-idx1-ubyte"))
    test_images = read_idx_images(os.path.join(raw_dir, "t10k-images-idx3-ubyte"))
    test_labels = read_idx_labels(os.path.join(raw_dir, "t10k-labels-idx1-ubyte"))
    return train_images, train_labels, test_images, test_labels


def build_pca_test_dataset(data_root, n_components=64):
    train_images, _, test_images, test_labels = load_mnist_raw(data_root)

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=43)
    train_features = pca.fit_transform(train_images).astype(np.float32)
    test_features = pca.transform(test_images).astype(np.float32)

    scale = np.max(np.abs(train_features)) + 1e-8
    test_features = test_features / scale
    return PCAMNISTDataset(test_features, test_labels)


def onn_outputs_to_logits(outputs, eps=1e-8):
    if outputs.dim() == 1:
        outputs = outputs.unsqueeze(0)
    outputs = outputs / (torch.amax(outputs, dim=1, keepdim=True) + eps)
    return F.interpolate(
        outputs.unsqueeze(1),
        size=10,
        mode="linear",
        align_corners=False,
    ).squeeze(1)


def add_output_noise(outputs, noise_amplitude, noise_type, generator=None):
    if noise_amplitude == 0:
        return outputs
    if noise_type == "gaussian":
        noise = torch.randn(
            outputs.shape,
            dtype=outputs.dtype,
            device=outputs.device,
            generator=generator,
        ) * noise_amplitude
    elif noise_type == "uniform":
        noise = (
            torch.rand(
                outputs.shape,
                dtype=outputs.dtype,
                device=outputs.device,
                generator=generator,
            )
            * 2
            - 1
        ) * noise_amplitude
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    return torch.clamp(outputs + noise, min=0.0)


def load_checkpoint_without_decay_buffers(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cuda")
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    filtered_state = {
        key: value
        for key, value in state_dict.items()
        if not key.endswith("energy_decay_power")
    }
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    decay_missing = [key for key in missing if key.endswith("energy_decay_power")]
    real_missing = [key for key in missing if not key.endswith("energy_decay_power")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {ckpt_path}: "
            f"missing={real_missing}, unexpected={unexpected}, decay_buffers={decay_missing}"
        )
    return checkpoint


def build_model(kind, physical_layers, args):
    if kind == "normal":
        return ONN_diff(
            layer_num=physical_layers,
            diffraction_size=args.diffraction_size,
            z=args.asm_z,
            pixelsize=args.asm_pixelsize,
            lamda=args.asm_lamda,
            energy_decay_db=args.normal_decay_db,
        )
    if physical_layers % 2 != 0:
        raise ValueError("Distributed diffraction physical layer count must be even.")
    return ONN_multi(
        layer_num=physical_layers // 2,
        diffraction_size=args.diffraction_size,
        z=args.asm_z,
        pixelsize=args.asm_pixelsize,
        lamda=args.asm_lamda,
        energy_decay_db=args.distri_decay_db,
    )


def checkpoint_path(root, kind, physical_layers, ckpt_name):
    if kind == "normal":
        exp_name = "mnist_diffraction"
    else:
        exp_name = "mnist_distributed_diffraction"
    return os.path.join(root, "record", exp_name, f"layers_{physical_layers}", "ckpt", ckpt_name)


def evaluate(model, testloader, args):
    model.eval()
    correct = 0
    total = 0
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)

    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Evaluating", leave=False):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(inputs)
            outputs = add_output_noise(
                outputs,
                noise_amplitude=args.noise_amplitude,
                noise_type=args.noise_type,
                generator=generator,
            )
            logits = onn_outputs_to_logits(outputs)
            predicted = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def plot_results(results, output_path, noise_amplitude, noise_type):
    layers = sorted({row["layers"] for row in results})
    accuracy_by_kind_layer = {
        (row["kind"], row["layers"]): row["accuracy"]
        for row in results
    }
    normal = [accuracy_by_kind_layer[("normal", layer)] for layer in layers]
    distri = [accuracy_by_kind_layer[("distributed", layer)] for layer in layers]

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(layers, normal, marker="o", linewidth=2, label="Normal diffraction, 12 dB/layer")
    plt.plot(layers, distri, marker="s", linewidth=2, label="Distributed diffraction, 8 dB/layer")
    plt.xlabel("Layer number")
    plt.ylabel("Test accuracy")
    plt.title(f"MNIST accuracy with {noise_type} output noise, amplitude={noise_amplitude:g}")
    plt.xticks(layers)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_csv(results, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "kind",
                "layers",
                "accuracy",
                "decay_db_per_model_layer",
                "checkpoint",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained MNIST diffraction models with output noise and real attenuation."
    )
    parser.add_argument("--noise-amplitude", type=float, default=0.0000001, help="Final-output noise amplitude.")
    parser.add_argument(
        "--noise-type",
        choices=["gaussian", "uniform"],
        default="gaussian",
        help="Gaussian uses amplitude as std; uniform samples from [-amplitude, amplitude].",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    parser.add_argument("--normal-decay-db", type=float, default=12.0)
    parser.add_argument("--distri-decay-db", type=float, default=8.0)
    parser.add_argument("--ckpt-name", default="ckpt_best.pth")
    parser.add_argument("--data-root", default="assets/attention/data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--asm-z", type=float, default=32 * 1e-6)
    parser.add_argument("--asm-pixelsize", type=float, default=50 * 1e-3)
    parser.add_argument("--asm-lamda", type=float, default=1550)
    parser.add_argument("--diffraction-size", type=int, default=640)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation script needs CUDA because the model classes allocate CUDA tensors.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(root, "record", "mnist_noise_layer_compare")
    os.makedirs(output_dir, exist_ok=True)

    test_dataset = build_pca_test_dataset(args.data_root, n_components=64)
    testloader = DataLoader(
        test_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    results = []
    for physical_layers in args.layers:
        for kind in ["normal", "distributed"]:
            ckpt_path = checkpoint_path(root, kind, physical_layers, args.ckpt_name)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

            model = build_model(kind, physical_layers, args).cuda()
            load_checkpoint_without_decay_buffers(model, ckpt_path)
            accuracy = evaluate(model, testloader, args)
            del model
            torch.cuda.empty_cache()

            decay = args.normal_decay_db if kind == "normal" else args.distri_decay_db
            row = {
                "kind": kind,
                "layers": physical_layers,
                "accuracy": accuracy,
                "decay_db_per_model_layer": decay,
                "checkpoint": ckpt_path,
            }
            results.append(row)
            print(f"{kind:11s} layers={physical_layers:2d} acc={accuracy:.4f} ckpt={ckpt_path}")

    noise_tag = f"{args.noise_type}_{args.noise_amplitude:g}".replace(".", "p")
    csv_path = os.path.join(output_dir, f"accuracy_vs_layers_{noise_tag}.csv")
    fig_path = os.path.join(output_dir, f"accuracy_vs_layers_{noise_tag}.png")
    save_csv(results, csv_path)
    plot_results(results, fig_path, args.noise_amplitude, args.noise_type)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
