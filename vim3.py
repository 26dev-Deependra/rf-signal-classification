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
            if not os.path.isdir(class_dir):
                print(f"Warning: Skipping {class_dir}, not a directory.")
                continue

            for fname in os.listdir(class_dir):
                path = os.path.join(class_dir, fname)
                # Basic check for image files
                if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.images.append((path, self.class_to_idx[cls]))
                else:
                    print(f"Warning: Skipping non-image file: {path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image and label
            return torch.randn(3, 128, 128), -1

        if self.transform:
            image = self.transform(image)

        # Handle cases where image loading failed
        if label == -1:
            return image, torch.tensor(-1)  # Return dummy tensor

        return image, label


# ==============================
# 2. Model
# ==============================
class VimForClassification(nn.Module):
    # ** MODIFIED ** Added more parameters to the init
    def __init__(self, num_classes, image_size=224, patch_size=16, dim=128,
                 depth=6, heads=4, dt_rank=16, dim_inner=None, d_state=None, channels=3, dropout=0.2):

        super().__init__()

        # Set defaults if None
        dim_inner = dim_inner if dim_inner is not None else dim
        d_state = d_state if d_state is not None else dim

        self.vim = Vim(
            dim=dim,
            heads=heads,
            dt_rank=dt_rank,
            dim_inner=dim_inner,
            d_state=d_state,
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
        # Handle dummy images from loading errors
        if -1 in labels:
            images = images[labels != -1]
            labels = labels[labels != -1]
            if images.size(0) == 0:
                continue

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    if len(loader.dataset) == 0:
        return 0.0  # Avoid division by zero
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            # Handle dummy images from loading errors
            if -1 in labels:
                images = images[labels != -1]
                labels = labels[labels != -1]
                if images.size(0) == 0:
                    continue

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    if len(loader.dataset) == 0 or total == 0:
        return 0.0, 0.0, [], []  # Avoid division by zero

    acc = correct / total
    return total_loss / len(loader.dataset), acc, labels_all, preds_all


# ==============================
# 4. Main
# ==============================
def main():
    # --- Config ---
    dataset_root = 'dataset'  # <-- Make sure this folder exists
    results_dir = 'vim_results_lightweight'  # <-- Changed output dir
    os.makedirs(results_dir, exist_ok=True)

    # --- ** MODIFIED HYPERPARAMETERS ** ---
    batch_size = 32        # Was 16
    num_epochs = 50        # Was 20
    lr = 1e-4              # Was 5e-4
    patience = 10          # Was 5
    weight_decay = 1e-3    # Was 1e-4
    # --- ** END MODIFICATIONS ** ---

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # --- ** MODIFIED DATA AUGMENTATION ** ---
    # More aggressive augmentation for 128x128 images
    transform = transforms.Compose([
        # Use RandomResizedCrop for more variety
        transforms.RandomResizedCrop((128, 128), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # Slightly more rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),

        # TrivialAugment is a strong, modern augmentation technique
        transforms.TrivialAugmentWide(),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    # --- ** END MODIFICATIONS ** ---

    dataset = CustomDataset(dataset_root, transform)
    if len(dataset) == 0:
        print(
            f"Error: No images found in {dataset_root}. Please check the path.")
        return

    num_classes = len(dataset.classes)
    print(
        f"Detected {len(dataset)} images in {num_classes} classes: {dataset.classes}")

    # --- Split ---
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    # --- ** MODIFIED MODEL INITIALIZATION ** ---
    print("Initializing Lightweight Vision Mamba (Vim)...")
    model = VimForClassification(
        num_classes=num_classes,
        image_size=128,      # <-- Match your image size
        patch_size=16,       # (128/16 = 8) -> 8x8 = 64 patches
        dim=64,              # <-- WAS 128: Drastically reduces parameter count
        depth=4,             # <-- WAS 6: Fewer layers
        heads=2,             # <-- WAS 4: Fewer heads
        dt_rank=8,           # <-- WAS 16: Smaller Mamba state rank
        dropout=0.4          # <-- WAS 0.2: Increase dropout
    ).to(device)
    # --- ** END MODIFICATIONS ** ---

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay  # <-- Use modified weight_decay
    )

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
                print(
                    f"Early stopping triggered after {patience} epochs with no improvement.")
                break

    # Check if any evaluation was actually run
    if not true_labels or not preds:
        print("Evaluation did not run. Skipping plots and metrics.")
        print(f"\nAll results saved under: {os.path.abspath(results_dir)}")
        return

    # --- Save Loss Plot ---
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss (Lightweight Vim)")
    loss_plot_path = os.path.join(results_dir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    print(f"Saved: {loss_plot_path}")
    plt.close()

    # --- Confusion Matrix ---
    cm = confusion_matrix(true_labels, preds)
    cm_percent = cm.astype(float) / cm.sum(axis=1)[:, None] * 100

    plt.figure(figsize=(max(6, num_classes), max(5, num_classes * 0.8)))
    plt.imshow(cm_percent, cmap='Blues',
               interpolation='nearest', vmin=0, vmax=100)
    plt.title("Confusion Matrix (%)")
    plt.colorbar()
    ticks = np.arange(num_classes)
    plt.xticks(ticks, dataset.classes, rotation=45, ha="right")
    plt.yticks(ticks, dataset.classes)

    # Text color threshold
    threshold = cm_percent.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)",
                     ha='center', va='center',
                     color="white" if cm_percent[i,
                                                 j] > threshold else "black",
                     fontsize=8)

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
