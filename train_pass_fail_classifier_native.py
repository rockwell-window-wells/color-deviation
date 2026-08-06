"""
Train a direct PASS/FAIL classifier from native-resolution part images
(no downsampling), so the model has access to defect area/extent, edge
smoothness, and shape - not just color - the way a human inspector
comparing against the boundary sample implicitly considers all of these,
without a fixed rule for which one(s) matter.

Trains on the "label" column directly (PASS/FAIL), not dE_max - this
matches the actual end goal (predict what a human would call) rather than
the color-difference proxy.

WHY THIS SCRIPT PADS INSTEAD OF REQUIRING UNIFORM IMAGE SIZES
    Native-resolution crops may not be identical in size across your
    well_size groups (depends on whether your custom CROP_BOXES share the
    same dimensions). Standard PyTorch batching requires every image in a
    batch to be the same shape. Rather than assume uniformity, this script
    pads every image in a batch up to that batch's own max height/width
    with neutral gray - so it works whether your crops are already uniform
    (in which case padding does nothing) or not.

MEMORY WARNING
    Native crops (e.g. 2000x1000) are far larger than the 224-512px range
    typically used for transfer learning, and GPU memory scales with image
    area. Default batch_size is small (2) for this reason. If you hit an
    out-of-memory error, reduce --batch_size further before reducing crop
    size - the whole point of this script is to avoid downsampling away
    the exact information (extent, smoothness, shape) you want to keep.
    Freezing the backbone (the default here) helps a lot: with all backbone
    weights frozen, PyTorch doesn't need to store their intermediate
    activations for backprop, which is the main memory cost at high
    resolution - --unfreeze_backbone will use dramatically more memory.

Usage:
    python train_pass_fail_classifier_native.py --manifest preprocessed_images_native/manifest.csv --epochs 20

Requirements:
    pip install torch torchvision pandas numpy scikit-learn pillow --break-system-packages
"""

import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PAD_VALUE = 0.5  # mid-gray in [0,1] range, applied before normalization


class NativeResPassFailDataset(Dataset):
    """Loads images at whatever resolution they're saved at - no resize.
    Only light geometric augmentation (flip) is safe to do without a fixed
    target size; heavier augmentation happens at collate time isn't
    practical here, so this stays minimal by design."""

    def __init__(self, df, train=True):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_filepath"]).convert("RGB")
        if self.train and np.random.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        tensor = transforms.functional.to_tensor(image)  # C,H,W in [0,1], no resize
        label = 1 if str(row["label"]).upper() == "FAIL" else 0
        return tensor, label


def pad_collate(batch):
    """Pads every image in the batch to that batch's own max H/W with
    neutral gray, then stacks into a single tensor. Runs on raw [0,1]
    tensors - normalization happens afterward on the whole padded batch,
    so the padding value is meaningful before it gets shifted/scaled."""
    images, labels = zip(*batch)
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded = []
    for img in images:
        c, h, w = img.shape
        canvas = torch.full((c, max_h, max_w), PAD_VALUE)
        top = (max_h - h) // 2
        left = (max_w - w) // 2
        canvas[:, top:top + h, left:left + w] = img
        padded.append(canvas)

    batch_tensor = torch.stack(padded)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return batch_tensor, labels_tensor


def normalize_batch(batch_tensor, device):
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (batch_tensor.to(device) - mean) / std


def build_model(freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Adaptive avg pool at the end of resnet already handles variable
    # spatial input size, so no architecture change needed for this.
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = normalize_batch(images, device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(fail)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def train_one_split(train_df, val_df, device, epochs, lr, batch_size, class_weights,
                     freeze_backbone=True):
    train_ds = NativeResPassFailDataset(train_df, train=True)
    val_ds = NativeResPassFailDataset(val_df, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, collate_fn=pad_collate)

    model = build_model(freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_auc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = normalize_batch(images, device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_ds)

        preds, labels_arr, probs = evaluate(model, val_loader, device)
        val_auc = roc_auc_score(labels_arr, probs) if len(np.unique(labels_arr)) > 1 else float("nan")
        scheduler.step(val_auc if not np.isnan(val_auc) else 0.0)

        print(f"  Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | val_AUC={val_auc:.4f}")

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    preds, labels_arr, probs = evaluate(model, val_loader, device)
    return model, {"preds": preds, "labels": labels_arr, "probs": probs, "auc": best_auc}


def print_fold_report(metrics, label=""):
    n_fail = int(metrics["labels"].sum())
    print(f"\n  Classification report{(' - ' + label) if label else ''} "
          f"(true fails: {n_fail} of {len(metrics['labels'])}):")
    if n_fail == 0 or n_fail == len(metrics["labels"]):
        print("  (Only one class present in this val set - skipping report)")
        return
    print(classification_report(metrics["labels"], metrics["preds"],
                                 target_names=["pass", "fail"], zero_division=0))
    print("  Confusion matrix (rows=true, cols=predicted), order: [pass, fail]")
    print(confusion_matrix(metrics["labels"], metrics["preds"]))


def compute_class_weights(labels):
    counts = np.bincount(labels, minlength=2)
    total = counts.sum()
    weights = total / (2 * counts)
    return torch.tensor(weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True,
                         help="manifest.csv from preprocess_thermal_images.py --no_resize")
    parser.add_argument("--group_col", default="truck_number")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2,
                         help="Kept small by default - native-resolution images are memory-"
                              "hungry. Reduce further if you hit out-of-memory errors.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unfreeze_backbone", action="store_true",
                         help="Uses substantially more GPU memory at native resolution - see "
                              "the memory warning in this file's docstring")
    parser.add_argument("--output", default="pass_fail_native_model.pt")
    parser.add_argument("--random_split_diagnostic", action="store_true",
                         help="Also run a plain random K-fold (not grouped by truck) as a "
                              "diagnostic only - see train_thermal_severity_regressor.py for "
                              "the full explanation of what this is and isn't evidence of.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(args.manifest)
    print(f"Loaded {len(df)} rows from {args.manifest}")

    if args.group_col in df.columns:
        n_missing = df[args.group_col].isna().sum()
        if n_missing > 0:
            print(f"{n_missing} row(s) missing '{args.group_col}' - excluding from grouped CV")
            df = df[df[args.group_col].notna()].reset_index(drop=True)

    groups = df[args.group_col].values
    n_folds = min(args.n_folds, len(np.unique(groups)))

    full_labels = (df["label"].str.upper() == "FAIL").astype(int).values
    class_weights = compute_class_weights(full_labels)
    print(f"Class balance: {int((full_labels==0).sum())} pass, {int((full_labels==1).sum())} fail | "
          f"loss weights: {class_weights.tolist()}")

    print(f"\n=== Cross-validation ({n_folds} folds, grouped by {args.group_col}) ===")
    gkf = GroupKFold(n_splits=n_folds)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        val_trucks = sorted(val_df[args.group_col].unique())
        print(f"\n--- Fold {fold + 1}/{n_folds} (val {args.group_col}s: {val_trucks}) ---")

        _, metrics = train_one_split(train_df, val_df, device, args.epochs, args.lr,
                                      args.batch_size, class_weights,
                                      freeze_backbone=not args.unfreeze_backbone)
        print(f"Fold {fold + 1} best val AUC: {metrics['auc']:.4f}")
        print_fold_report(metrics, label=f"fold {fold + 1}")
        fold_results.append(metrics)

    print(f"\n=== Cross-validation summary ===")
    aucs = [m["auc"] for m in fold_results]
    print(f"AUC: {np.mean(aucs):.4f} (std {np.std(aucs):.4f})")

    pooled_preds = np.concatenate([m["preds"] for m in fold_results])
    pooled_labels = np.concatenate([m["labels"] for m in fold_results])
    pooled_metrics = {"preds": pooled_preds, "labels": pooled_labels}
    print("\n=== Pooled classification report across all CV folds ===")
    print_fold_report(pooled_metrics)

    if args.random_split_diagnostic:
        print(f"\n{'='*70}")
        print(f"=== DIAGNOSTIC: plain random {n_folds}-fold (NOT grouped by truck) ===")
        print(f"{'='*70}")
        print("Same caveat as the regression script: this can leak truck identity across")
        print("train/val, so it's expected to look better even if the model hasn't learned")
        print("anything generalizable. Only useful for the yes/no question: can the model")
        print("fit the pattern at all when given same-truck examples on both sides?")

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        random_aucs = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            print(f"\n--- Random fold {fold + 1}/{n_folds} ---")
            _, metrics = train_one_split(train_df, val_df, device, args.epochs, args.lr,
                                          args.batch_size, class_weights,
                                          freeze_backbone=not args.unfreeze_backbone)
            print(f"Random fold {fold + 1} best val AUC: {metrics['auc']:.4f}")
            random_aucs.append(metrics["auc"])

        print(f"\n=== Random-split diagnostic summary ===")
        print(f"AUC: {np.mean(random_aucs):.4f} (std {np.std(random_aucs):.4f})")
        print(f"Compare to grouped CV AUC: {np.mean(aucs):.4f}")

    print(f"\n=== Training final deployment model on ALL {len(df)} labeled images ===")
    print("(Note: the val_AUC printed during THIS run overlaps with training data - not a "
          "real validation metric. Use the CV summary above for the honest estimate.)")
    final_model, _ = train_one_split(
        df, df.sample(min(20, len(df)), random_state=42), device, args.epochs, args.lr,
        args.batch_size, class_weights, freeze_backbone=not args.unfreeze_backbone,
    )

    torch.save({"model_state_dict": final_model.state_dict()}, args.output)
    print(f"\nSaved final model to {args.output}")
    print(f"Expected real-world performance (from cross-validation): AUC={np.mean(aucs):.4f}")


if __name__ == "__main__":
    main()
