"""
Train a model to recognize well_size directly from the part image - both to
drive automatic size-aware cropping (no reliance on possibly-mistyped
metadata at inference time), and to AUDIT your existing labeled dataset for
rows where the recorded well_size doesn't match what the image actually
shows (a systematic way to catch the kind of mislabeling you found by hand).

Given how visually distinct the four sizes are (framing/zoom level differs
a lot - e.g. X24 parts show much more background/ceiling than X38 parts),
this should be a much easier, higher-accuracy task than the severity
regression problem.

Usage:
    python train_well_size_classifier.py --manifest training_set/manifest.csv --epochs 15

Requirements:
    pip install torch torchvision pandas numpy scikit-learn pillow --break-system-packages
"""

import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class WellSizeDataset(Dataset):
    def __init__(self, df, class_to_idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_filepath"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[int(row["well_size"])]
        return image, label


def get_transforms(img_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(),
            # No color augmentation here at all - this task is about framing/
            # geometry, not color, and we want to leave color info untouched
            # in case we reuse this backbone for anything color-related later.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def build_model(num_classes, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_confidences = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_confidences.extend(confs.cpu().tolist())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return acc, all_preds, all_labels, all_confidences


def train_one_split(train_df, val_df, class_to_idx, device, img_size, epochs, lr, batch_size):
    train_ds = WellSizeDataset(train_df, class_to_idx, get_transforms(img_size, train=True))
    val_ds = WellSizeDataset(val_df, class_to_idx, get_transforms(img_size, train=False))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model(len(class_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_ds)

        val_acc, _, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_acc)
        print(f"  Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model, best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--group_col", default="truck_number")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="well_size_model.pt")
    parser.add_argument("--mismatch_report", default="well_size_mismatches.csv",
                         help="Where to save rows where the recorded well_size disagrees with "
                              "what the trained model sees in the image")
    parser.add_argument("--mismatch_confidence_min", type=float, default=0.9,
                         help="Only flag mismatches where the model is at least this confident, "
                              "to avoid flagging genuinely ambiguous/borderline images")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(args.manifest)
    print(f"Loaded {len(df)} rows from {args.manifest}")

    if args.group_col in df.columns:
        n_missing = df[args.group_col].isna().sum()
        if n_missing > 0:
            print(f"{n_missing} row(s) missing '{args.group_col}' - excluding from CV (still used in final model)")
        cv_df = df[df[args.group_col].notna()].reset_index(drop=True)
    else:
        cv_df = df

    classes = sorted(df["well_size"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    print(f"Classes: {classes}")

    if args.group_col in cv_df.columns and cv_df[args.group_col].notna().all():
        groups = cv_df[args.group_col].values
        n_folds = min(args.n_folds, len(np.unique(groups)))
        print(f"\n=== Cross-validation ({n_folds} folds, grouped by {args.group_col}) ===")
        gkf = GroupKFold(n_splits=n_folds)
        accs = []
        all_cv_preds, all_cv_labels = [], []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(cv_df, groups=groups)):
            train_df = cv_df.iloc[train_idx]
            val_df = cv_df.iloc[val_idx]
            print(f"\n--- Fold {fold + 1}/{n_folds} ---")
            model, best_acc = train_one_split(train_df, val_df, class_to_idx, device,
                                               args.img_size, args.epochs, args.lr, args.batch_size)
            print(f"Fold {fold + 1} best val accuracy: {best_acc:.4f}")

            val_loader = DataLoader(WellSizeDataset(val_df, class_to_idx, get_transforms(args.img_size, False)),
                                     batch_size=args.batch_size, shuffle=False)
            _, preds, labels, _ = evaluate(model, val_loader, device)
            all_cv_preds.extend(preds)
            all_cv_labels.extend(labels)
            accs.append(best_acc)

        print(f"\n=== Cross-validation summary ===")
        print(f"Mean accuracy: {np.mean(accs):.4f} (std {np.std(accs):.4f})")
        target_names = [str(idx_to_class[i]) for i in range(len(classes))]
        print("\nPooled classification report:")
        print(classification_report(all_cv_labels, all_cv_preds, target_names=target_names))
        print("Confusion matrix (rows=true, cols=predicted), order:", target_names)
        print(confusion_matrix(all_cv_labels, all_cv_preds))

    print(f"\n=== Training final model on ALL {len(df)} images ===")
    final_model, _ = train_one_split(
        df, df.sample(min(30, len(df)), random_state=42),
        class_to_idx, device, args.img_size, args.epochs, args.lr, args.batch_size,
    )

    torch.save({
        "model_state_dict": final_model.state_dict(),
        "class_to_idx": class_to_idx,
        "img_size": args.img_size,
    }, args.output)
    print(f"Saved model to {args.output}")

    # Audit pass: run the final model over every image and flag any row
    # where the recorded well_size disagrees with what the model sees,
    # at high confidence - a systematic check for the kind of mislabeling
    # you already found by hand.
    print(f"\n=== Auditing all {len(df)} images for well_size mismatches ===")
    full_loader = DataLoader(WellSizeDataset(df, class_to_idx, get_transforms(args.img_size, False)),
                              batch_size=args.batch_size, shuffle=False)
    _, preds, labels, confidences = evaluate(final_model, full_loader, device)

    df = df.reset_index(drop=True)
    df["predicted_well_size"] = [idx_to_class[p] for p in preds]
    df["prediction_confidence"] = confidences
    mismatches = df[
        (df["predicted_well_size"] != df["well_size"])
        & (df["prediction_confidence"] >= args.mismatch_confidence_min)
    ]

    if len(mismatches) > 0:
        cols = ["part_id", "image_filepath", "well_size", "predicted_well_size", "prediction_confidence"]
        cols = [c for c in cols if c in mismatches.columns]
        mismatches[cols].sort_values("prediction_confidence", ascending=False).to_csv(
            args.mismatch_report, index=False
        )
        print(f"{len(mismatches)} likely mislabeled row(s) found (model disagrees with recorded "
              f"well_size at >= {args.mismatch_confidence_min:.0%} confidence)")
        print(f"Details written to {args.mismatch_report} - worth manually checking these images")
        print(mismatches[cols].head(10).to_string(index=False))
    else:
        print("No high-confidence mismatches found.")


if __name__ == "__main__":
    main()
