import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import datetime

# ==============================
# Import Vision Mamba
# ==============================
try:
    from vision_mamba import Vim
except ImportError:
    raise ImportError("Install Vision Mamba: pip install vision-mamba")


# ==============================
# 1. Dataset
# ==============================
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(class_dir):
                path = os.path.join(class_dir, fname)
                self.images.append((path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ==============================
# 2. Model
# ==============================
class VimForClassification(nn.Module):
    def __init__(self, num_classes, image_size=224, patch_size=16, dim=128, depth=6, channels=3, dropout=0.2):
        super().__init__()
        self.vim = Vim(
            dim=dim,
            heads=4,
            dt_rank=16,
            dim_inner=dim,
            d_state=dim,
            num_classes=num_classes,
            image_size=image_size,
            patch_size=patch_size,
            channels=channels,
            dropout=dropout,
            depth=depth
        )

    def forward(self, x):
        return self.vim(x)


# ==============================
# 3. Train & Evaluate
# ==============================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    acc = correct / total
    return total_loss / len(loader.dataset), acc, labels_all, preds_all


# ==============================
# 4. Main
# ==============================
def main():
    # --- Config ---
    dataset_root = 'dataset'
    results_dir = 'vim_results'
    os.makedirs(results_dir, exist_ok=True)

    batch_size = 16
    num_epochs = 20
    lr = 5e-4
    patience = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Using device:", device)

    # --- Data Augmentation ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    dataset = CustomDataset(dataset_root, transform)
    num_classes = len(dataset.classes)
    print(f"Detected classes: {dataset.classes}")

    # --- Split ---
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # --- Model ---
    model = VimForClassification(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # --- Training Loop ---
    best_acc = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    metrics_log = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, true_labels, preds = evaluate(
            model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        metrics_log.append((epoch+1, train_loss, val_loss, val_acc))

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(
                results_dir, "best_vim_model.pth"))
            print("** Saved new best model **")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # --- Save Loss Plot ---
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    loss_plot_path = os.path.join(results_dir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    print(f"Saved: {loss_plot_path}")
    plt.close()

    # --- Confusion Matrix ---
    cm = confusion_matrix(true_labels, preds)
    cm_percent = cm.astype(float) / cm.sum(axis=1)[:, None] * 100
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_percent, cmap='Blues', interpolation='nearest')
    plt.title("Confusion Matrix (%)")
    plt.colorbar()
    ticks = np.arange(num_classes)
    plt.xticks(ticks, dataset.classes, rotation=45)
    plt.yticks(ticks, dataset.classes)
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, f"{cm[i, j]} ({cm_percent[i, j]:.1f}%)",
                     ha='center', va='center',
                     color="white" if cm_percent[i, j] > 50 else "black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Saved: {cm_path}")
    plt.close()

    # --- Save Metrics Log ---
    metrics_path = os.path.join(results_dir, "training_metrics.csv")
    with open(metrics_path, "w") as f:
        f.write("Epoch,Train_Loss,Val_Loss,Val_Accuracy\n")
        for e, tr, vl, acc in metrics_log:
            f.write(f"{e},{tr:.6f},{vl:.6f},{acc:.6f}\n")
    print(f"Saved: {metrics_path}")

    print(f"\nAll results saved under: {os.path.abspath(results_dir)}")


if __name__ == "__main__":
    main()
