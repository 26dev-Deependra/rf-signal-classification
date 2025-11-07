import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torchsummary import summary
from torchviz import make_dot

# Import Vim model
try:
    from vision_mamba import Vim
except ImportError:
    raise ImportError(
        "You must install the vision-mamba package (pip install vision-mamba)")

# ================================================================
# 1. Dataset Definition
# ================================================================


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


# ================================================================
# 2. Model Definition using Vision Mamba
# dim = 256 each patch size,
# ================================================================
class VimForClassification(nn.Module):
    def __init__(self, num_classes, image_size=224, patch_size=16, dim=256, depth=12, channels=3, dropout=0.1):
        super().__init__()
        self.vim = Vim(
            dim=dim,
            heads=8,
            dt_rank=32,
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


# ================================================================
# 3. Training & Evaluation Helpers
# ================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = correct / total if total > 0 else 0
    return running_loss / len(loader.dataset), accuracy, all_labels, all_preds


# ================================================================
# 4. Main execution
# ================================================================
def main():
    # ----- Configuration -----
    dataset_root = 'dataset'
    batch_size = 8
    num_epochs = 10
    lr = 1e-4
    image_size = 224
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # ----- Transformations & Data Loaders -----
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    dataset = CustomDataset(root_dir=dataset_root, transform=transform)
    num_classes = len(dataset.classes)
    print("Detected classes:", dataset.classes)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # ----- Model, Loss, Optimizer -----
    model = VimForClassification(
        num_classes=num_classes, image_size=image_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ----- Training Loop -----
    train_losses = []
    val_losses = []
    best_acc = 0.0
    best_model_path = 'best_vim_model.pth'

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        t_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, true_labels, preds = evaluate(
            model, val_loader, criterion, device)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        print(
            f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}")

        # Save best model
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), best_model_path)
            print("** Saved best model **")

    # ----- Loss Plot -----
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses,
             label='Train Loss', color='blue')
    plt.plot(range(1, num_epochs+1), val_losses,
             label='Val Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.show()

    # ----- Confusion Matrix -----
    cm = confusion_matrix(true_labels, preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, None] * 100
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (%)')
    plt.colorbar()
    ticks = np.arange(num_classes)
    plt.xticks(ticks, dataset.classes, rotation=45)
    plt.yticks(ticks, dataset.classes)
    thresh = cm_percent.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, f'{cm[i, j]} ({cm_percent[i, j]:.1f}%)',
                     ha="center", color="white" if cm_percent[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # ----- Model Summary & Graph -----
    summary(model, input_size=(3, image_size, image_size))
    dummy = torch.zeros((1, 3, image_size, image_size)).to(device)
    vis_graph = make_dot(model(dummy), params=dict(model.named_parameters()))
    vis_graph.render('vim_model_architecture', format='png')
    print("Model architecture image saved: vim_model_architecture.png")


if __name__ == '__main__':
    main()
