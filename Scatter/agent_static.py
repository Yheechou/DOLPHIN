import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.scatter_static import Scatter_static


exp_name = 'static_scattering' 
object_size = [600, 600]
print(exp_name)
root = "assets/scattering"
os.makedirs(os.path.join(root ,"record", exp_name),exist_ok=True)

traindir = 'data/traffic/Traffic_signs_1000/Train'
testdir = 'data/traffic/Traffic_signs_1000/Test'
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((object_size[0], object_size[1])),
    transforms.ToTensor(),  
])

train_dataset = datasets.ImageFolder(traindir, transform)
test_dataset = datasets.ImageFolder(testdir, transform)

trainloader = DataLoader(train_dataset, num_workers=1, batch_size=16, shuffle=True)
testloader = DataLoader(test_dataset, num_workers=1, batch_size=16, shuffle=False)

model = Scatter_static(object_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

model = model.cuda()
criterion = nn.CrossEntropyLoss()

load_ckpt = True
if load_ckpt:
    checkpoint = torch.load('assets/scattering/record/static_0.8pi/ckpt.pth')
    model.load_state_dict(checkpoint['model_state'], strict=False)
    model.encoder_phase.requires_grad = False
    model.decoder_phase.requires_grad = False


for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)

file_path = os.path.join(root, "record", exp_name, "record.txt")
with open(file_path, "w", encoding="utf-8") as file:
    file.write(f"exp_name:{exp_name}\n")

num_epochs = 80
best_accuracy = 0.
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f"Epoch [{epoch+1}/{num_epochs}]") as pbar:
        for i, (objects, labels) in pbar:
            optimizer.zero_grad()
            objects, labels = objects.cuda(), labels.cuda()
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
                "Running Accuracy": f"{100 * running_correct / running_total:.2f}%"
            })

    accuracy = 100 * running_correct / running_total
    average_loss = running_loss / len(trainloader)
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(f'Train epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.2f}%, Average Loss: {average_loss:.4f}   ')

    # Evaluate the model after each epoch
    model.eval()
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
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(f'Test epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.2f}%, Average Loss: {average_loss:.4f}\n')
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        state = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(state, os.path.join(root, 'record', exp_name, 'ckpt.pth'))
