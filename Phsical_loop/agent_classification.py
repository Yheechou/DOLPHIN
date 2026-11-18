import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils.classification import Classification
from utils.plot import plot_img
from utils.utils import print_trainable_parameters, create_circle_image, create_square_image, create_square_image_noise



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=['traffic', 'mnist', 'fashion_mnist'],
        default='mnist',
        help='Choose dataset: traffic signs, MNIST, or Fashion-MNIST.',
    )
    return parser.parse_args()


args = parse_args()

exp_name = "mnist_classification"
root = "assets/loop"
object_size = [600, 600]
batch_size = 32
lr = 1e-3
device = 'cuda'

os.makedirs(os.path.join(root, "record_classification", exp_name),exist_ok=True)
print(exp_name)
file_path = os.path.join(root, "record_classification", exp_name, "record.txt")

# dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(object_size[0]),
    transforms.ToTensor(),  
])

traindir = 'data/traffic/Traffic_signs_1000/Train'
testdir = 'data/traffic/Traffic_signs_1000/Test'

if args.dataset == 'traffic':
    train_dataset = datasets.ImageFolder(traindir, transform)
    test_dataset = datasets.ImageFolder(testdir, transform)
    num_classes = len(train_dataset.classes)
elif args.dataset == 'mnist':
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    num_classes = 10
elif args.dataset == 'fashion_mnist':
    train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    num_classes = 10
else:
    raise ValueError(f"Unsupported dataset: {args.dataset}")

with open(file_path, "a", encoding="utf-8") as file:
    file.write(f"num_classes:{num_classes}\n")

trainloader = DataLoader(train_dataset, num_workers=1, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, num_workers=1, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
model = Classification(object_size=object_size, num_classes=num_classes)
model = model.to(device)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# load checkpoint if needed
load_ckpt = True
if load_ckpt:
    checkpoint = torch.load(os.path.join(root, "record_classification", exp_name, 'ckpt.pth'))
    model.load_state_dict(checkpoint['model_state'], strict=False)

num_epochs = 80
best_loss = float('inf')
best_accuracy = 0.0

for epoch in range(num_epochs):




    
        # Evaluate the model after each epoch
    model.eval()
    correct = 0
    total = 0
    running_test_correct = 0
    running_test_total = 0
    running_test_loss = 0.0

    with torch.no_grad():
        with tqdm(enumerate(testloader, 0), total=len(testloader), desc="Evaluating", leave=False) as test_pbar:
            for i, (objects, labels) in test_pbar:
                objects, labels = objects.cuda(), labels.cuda()

                outputs = model(objects)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)

                running_test_total += labels.size(0)
                running_test_correct += (predicted == labels).sum().item()
                running_test_loss += loss.item()

                test_pbar.set_postfix({
                    "Test Accuracy": f"{100 * running_test_correct / running_test_total:.2f}%",
                    "Test Loss": f"{running_test_loss / (i + 1):.4f}"
                })

    accuracy = 100 * running_test_correct / running_test_total
    average_loss = running_test_loss / len(testloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.2f}%, Average Loss: {average_loss:.4f}')
    breakpoint()
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(f'Test epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.2f}%, Average Loss: {average_loss:.4f}\n')
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        state = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(state, os.path.join(root, "record_classification", exp_name, 'ckpt.pth'))
