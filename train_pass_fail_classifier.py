"""
Train a pass/fail binary image classifier using transfer learning (ResNet18).

Expected input:
    A folder of "pass" images and a folder of "fail" images, e.g.:
        /path/to/pass/*.jpg
        /path/to/fail/*.jpg

What this script does:
    1. Splits your images into train/val sets automatically (stratified by class).
    2. Applies data augmentation (helps a lot with small datasets).
    3. Fine-tunes a pretrained ResNet18 on your two classes.
    4. Tracks validation accuracy and saves the best-performing model weights.
    5. Prints a final classification report (precision/recall/F1) on the val set.

Usage:
    python train_pass_fail_classifier.py --pass_dir ./pass --fail_dir ./fail --epochs 20

Requirements:
    pip install torch torchvision scikit-learn pillow --break-system-packages
"""

import argparse
import copy
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_train_val_split(pass_dir, fail_dir, work_dir, val_fraction=0.2, seed=42):
    """
    torchvision's ImageFolder expects a directory structure like:
        work_dir/train/pass/*.jpg
        work_dir/train/fail/*.jpg
        work_dir/val/pass/*.jpg
        work_dir/val/fail/*.jpg

    This copies your images into that structure with a stratified split,
    so pass/fail images are never mixed between train and val.
    """
    random.seed(seed)
    work_dir = Path(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)

    for split in ("train", "val"):
        for cls in ("pass", "fail"):
            (work_dir / split / cls).mkdir(parents=True, exist_ok=True)

    for cls, src_dir in (("pass", pass_dir), ("fail", fail_dir)):
        files = [f for f in Path(src_dir).iterdir() if f.suffix.lower() in IMG_EXTENSIONS]
        if not files:
            raise ValueError(f"No images found in {src_dir}")
        random.shuffle(files)
        n_val = max(1, int(len(files) * val_fraction))
        val_files = files[:n_val]
        train_files = files[n_val:]

        for f in train_files:
            shutil.copy(f, work_dir / "train" / cls / f.name)
        for f in val_files:
            shutil.copy(f, work_dir / "val" / cls / f.name)

        print(f"{cls}: {len(train_files)} train, {len(val_files)} val")

    return work_dir


def get_dataloaders(work_dir, img_size=224, batch_size=16):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(Path(work_dir) / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(Path(work_dir) / "val", transform=val_tf)

    # class_to_idx is alphabetical: {'fail': 0, 'pass': 1}
    print("Class mapping:", train_ds.class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_ds.class_to_idx


def build_model(freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model


def compute_class_weights(train_ds, device):
    """
    Inverse-frequency class weights, so the minority class (fail) contributes
    proportionally more to the loss. With e.g. 200 pass / 70 fail, fail
    mistakes get weighted ~2.9x more heavily than pass mistakes.
    """
    counts = [0, 0]
    for _, label in train_ds.samples:
        counts[label] += 1

    total = sum(counts)
    weights = [total / (len(counts) * c) for c in counts]
    print(f"Class counts (idx order): {counts} -> loss weights: {[round(w, 3) for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_model(model, train_loader, val_loader, device, class_weights, epochs=20, lr=1e-3,
                 fail_idx=None):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Only optimize params that require grad (the new fc layer, if backbone frozen)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_score = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            total += inputs.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total

        val_acc, val_loss, val_preds, val_labels = evaluate(model, val_loader, device, criterion)

        # Recall on the fail class: of the actual fails, how many did we catch?
        fail_total = sum(1 for l in val_labels if l == fail_idx)
        fail_caught = sum(1 for p, l in zip(val_preds, val_labels) if l == fail_idx and p == fail_idx)
        fail_recall = fail_caught / fail_total if fail_total else 0.0

        scheduler.step(fail_recall)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_fail_recall={fail_recall:.4f}")

        # Select the checkpoint that catches the most fails, using overall
        # accuracy only as a tiebreaker (otherwise a model that just predicts
        # "fail" for everything would win on recall alone).
        score = fail_recall + 0.01 * val_acc
        if score > best_score:
            best_score = score
            best_weights = copy.deepcopy(model.state_dict())

    print(f"\nBest checkpoint selected (fail_recall + small accuracy tiebreak)")
    model.load_state_dict(best_weights)
    return model


def evaluate(model, loader, device, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    running_loss, total = 0.0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)

            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    avg_loss = running_loss / total if total else 0.0
    return acc, avg_loss, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass_dir", required=True, help="Folder containing pass images")
    parser.add_argument("--fail_dir", required=True, help="Folder containing fail images")
    parser.add_argument("--work_dir", default="./_split_data", help="Where to build the train/val split")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--unfreeze_backbone", action="store_true",
                         help="Fine-tune the whole network instead of just the final layer "
                              "(use if you have a larger dataset, e.g. 200+ images per class)")
    parser.add_argument("--output", default="pass_fail_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    work_dir = build_train_val_split(args.pass_dir, args.fail_dir, args.work_dir, args.val_fraction)
    train_loader, val_loader, class_to_idx = get_dataloaders(work_dir, batch_size=args.batch_size)
    fail_idx = class_to_idx["fail"]

    class_weights = compute_class_weights(train_loader.dataset, device)

    model = build_model(freeze_backbone=not args.unfreeze_backbone)
    model = train_model(model, train_loader, val_loader, device, class_weights,
                         epochs=args.epochs, lr=args.lr, fail_idx=fail_idx)

    # Final evaluation report
    _, _, preds, labels = evaluate(model, val_loader, device)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_names = [idx_to_class[i] for i in sorted(idx_to_class)]

    print("\nClassification report (validation set):")
    print(classification_report(labels, preds, target_names=target_names))
    print("Confusion matrix (rows=true, cols=predicted), order:", target_names)
    print(confusion_matrix(labels, preds))

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
    }, args.output)
    print(f"\nSaved model to {args.output}")


if __name__ == "__main__":
    main()
