# Dolphin

This repository contains the experiments accompanying the Dolphin paper. 

| Directory | Experiment |
| --- | --- |
| `Cityscape_detection&segmentation/` | Object detection, depth estimation and semantic segmentation on Cityscapes |
| `Distributed_diffraction_loss/` | Effects of diffraction loss and layer-specific noise on MNIST classification |
| `Drone_cruising_decisions/` | Training and evaluation of drone navigation decisions using AirSim data |
| `Phsical_loop/` | Classification performence with the number of physical loop |
| `Scatter/` | Static and dynamic scattering experiments and classification under scattering |

## Environment

- Python 3.12.9 and PyTorch 2.3.1 (`torch==2.3.1`)
- Additional packages include `torchvision`, `numpy`, `Pillow`, `opencv-python`, `scikit-image`, `scikit-learn`, `matplotlib`, and `tqdm`.
- The YOLO detection baseline also requires `ultralytics`.

Some scripts use CUDA by default. Before running an experiment, set the dataset path, checkpoint path, and compute device in the corresponding training or evaluation script.

## Data and checkpoints

This repository contains the Python source. Paths beginning with `data/` refer to datasets. Paths beginning with `assets/` refer to experiment-specific files such as calibration arrays, checkpoints, and training records. These relative paths resolve from the process working directory. Match the paths in the relevant script to the locations of your local files.

The hardware calibration scripts use `192.0.2.1` as an example device address. Set the address to match your instrument before use.