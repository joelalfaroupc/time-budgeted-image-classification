import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
if __name__ == '__main__':
    # ---------- Reproducibilitat ----------
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)

    # ---------- Transformacions i Augmentació ----------
    train_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3)
        ], p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # ---------- Dataset ----------
    data_dir = "./data/train"
    dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    class_names = dataset.classes
    num_classes = len(class_names)

    # Subset per avaluació entrenament
    eval_subset_size = 1000
    indices = random.sample(range(len(dataset)), eval_subset_size)
    train_eval_subset = Subset(dataset, indices)

    # Dataset de validació
    val_dir = "./data/validation"
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    # ---------- Model ----------
    class ElVostreModel(nn.Module):
        def __init__(self, num_classes):
            super(ElVostreModel, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(2)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(2)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(2)
            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.01),
                nn.AdaptiveAvgPool2d((1,1))  # global avg pool
            )
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(256, 256)
            self.fc2 = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(F.leaky_relu(self.fc1(x), negative_slope=0.01))
            x = self.fc2(x)
            return x

    # ---------- Entrenament ----------
    lr = 0.001
    batch_size = 32
    max_total_time = 600  # 10 minuts

    device = torch.device("cpu")
    model = ElVostreModel(num_classes).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, threshold=0.001, cooldown=1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    # ---------- Dataloaders ----------
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_eval_loader = DataLoader(train_eval_subset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ---------- Avaluació ----------
    def evaluate(loader, name):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            total_loss = 0.0
            total_batches = 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                total_batches += 1
            accuracy = correct / total
            avg_loss = total_loss / total_batches if total_batches > 0 else 0
            print(f"{name} Accuracy: {accuracy * 100:.2f}% | Avg Loss: {avg_loss:.4f}")
            return accuracy

    # ---------- Entrenament amb temps màxim ----------
    start_time = time.time()
    epoch = 0
    max_time = False

    best_val_acc = 0.0
    best_model_state = None

    while not max_time:
        epoch += 1
        model.train()
        loop = tqdm(dataloader, desc=f"Època {epoch}", leave=False)

        for images, labels in loop:
            if time.time() - start_time > max_total_time:
                print("Temps màxim assolit. Fi de l'entrenament")
                max_time = True
                break

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        scheduler.step(loss)

        # Validació per època
        val_acc = evaluate(val_loader, f"Validació (després de la època {epoch})")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    # ---------- Avaluació final ----------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best (Validation Accuracy: {best_val_acc * 100:.2f}%)")