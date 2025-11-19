"""
Pizza vs Sushi CNN: Spark + PyTorch
"""

import os
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from pyspark.sql import SparkSession

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class PizzaSushiNet(nn.Module):
    def __init__(self, num_classes: int):
        super(PizzaSushiNet, self).__init__()
        # Input: 3 x 128 x 128
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # After 3x pooling: 128 -> 64 -> 32 -> 16
        self.flatten_dim = 128 * 16 * 16

        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(-1, self.flatten_dim)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # logits


def train_one_epoch(model, device, loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Optional: print occasionally
        if (batch_idx + 1) % 10 == 0:
            print(
                f"[Epoch {epoch}] Batch {batch_idx+1}/{len(loader)} "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, device, loader, criterion, class_names, set_name="VAL"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = running_loss / total
    avg_acc = correct / total

    print("\n" + "*" * 60)
    print(f"{set_name} SET : METRICS (PyTorch)")
    print("-" * 60)
    print(f"{set_name} loss:     {avg_loss:.4f}")
    print(f"{set_name} accuracy: {avg_acc:.4f}")

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix ({set_name}):")
    print(cm)

    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print(f"\nPer-class accuracy ({set_name}):")
    for name, acc in zip(class_names, per_class_acc):
        print(f"  {name}: {acc:.4f}")

    print(f"\nClassification Report ({set_name}):")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return avg_loss, avg_acc, cm


def main():
    ###############################################################
    # 0. Initialize Spark (for spark-submit compliance)
    ###############################################################
    spark = SparkSession.builder \
        .appName("PizzaVsSushi_CNN_PyTorch") \
        .master("spark://hadoop1:7077") \
        .config("spark.executor.memory", "2g") \
        .config("spark.executor.cores", "2") \
        .getOrCreate()

    ###############################################################
    # 1. Paths & basic settings
    ###############################################################
    train_dir = Path("/home/sat3812/Downloads/proj4/small_project_4/train")
    test_dir = Path("/home/sat3812/Downloads/proj4/small_project_4/test")

    img_size = (128, 128)
    batch_size = 8
    seed = 42
    num_epochs = 10
    val_split = 0.2

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ###############################################################
    # 2. Transforms & Datasets (ImageFolder)
    ###############################################################
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    full_train_dataset = datasets.ImageFolder(root=str(train_dir),
                                              transform=train_transform)
    class_names = full_train_dataset.classes
    num_classes = len(class_names)
    print("Class names:", class_names)

    # train/validation split
    n_total = len(full_train_dataset)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    # For validation, we usually do not apply random augmentation
    val_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    # Test dataset
    test_dataset = datasets.ImageFolder(root=str(test_dir),
                                        transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    print("Test class names:", test_dataset.classes)

    ###############################################################
    # 3. Model, criterion, optimizer
    ###############################################################
    model = PizzaSushiNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    print(model)

    ###############################################################
    # 4. Training loop (with simple early stopping)
    ###############################################################
    best_val_loss = float("inf")
    best_state = None
    patience = 3
    patience_counter = 0

    history_train_acc = []
    history_val_acc = []

    for epoch in range(1, num_epochs + 1):
        print("\n" + "=" * 60)
        print(f"Epoch {epoch}/{num_epochs}")
        print("=" * 60)

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )
        val_loss, val_acc, _ = evaluate(
            model, device, val_loader, criterion, class_names, set_name="VAL"
        )
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch} done in {elapsed:.2f} sec")
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

        history_train_acc.append(train_acc)
        history_val_acc.append(val_acc)

        # Early stopping logic (on val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
            print(">> New best model (val_loss improved).")
        else:
            patience_counter += 1
            print(f">> No improvement in val_loss (patience={patience_counter}/{patience}).")
            if patience_counter >= patience:
                print(">> Early stopping triggered.")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    ###############################################################
    # 5. Final training metrics summary
    ###############################################################
    final_train_acc = history_train_acc[-1] if history_train_acc else 0.0
    final_val_acc = history_val_acc[-1] if history_val_acc else 0.0

    print("\n" + "*" * 60)
    print("FINAL TRAINING METRICS")
    print("-" * 60)
    print(f"Final training accuracy:   {final_train_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")

    ###############################################################
    # 6. Detailed validation metrics (again, for report)
    ###############################################################
    _ = evaluate(model, device, val_loader, criterion,
                 class_names, set_name="VAL (FINAL)")

    ###############################################################
    # 7. Test set evaluation + metrics
    ###############################################################
    ###############################################################
    # 7. Test set evaluation + metrics
    ###############################################################
    test_loss, test_acc, test_cm = evaluate(
        model,
        device,
        test_loader,
        criterion,
        class_names=test_dataset.classes,
        set_name="TEST"
    )

    print("\n" + "*" * 60)
    print("FINAL TEST METRICS SUMMARY")
    print("-" * 60)
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("*" * 60)

    print("\n" + "=" * 60)
    print("DONE.")
    print("=" * 60)

    ###############################################################
    # 8. Stop Spark session
    ###############################################################
    spark.stop()


if __name__ == "__main__":
    main()

