"""
Train a model to predict dE_max (thermal-shock color severity) directly
from a part image, using the manifest built by build_thermal_training_set.py.

WHY REGRESSION, NOT CLASSIFICATION
    Instead of training the model to output PASS/FAIL directly, it predicts
    the continuous severity number (dE_max) your color measurements already
    established as the right target. Pass/fail is then just "is the
    predicted dE_max above the threshold you already validated (3.21)" -
    same decision rule as the measurement-based process, just fed by a
    predicted number instead of a physically measured one. This also means
    if you ever want to retune the threshold, you don't need to retrain
    the model - just change the cutoff.

WHY THE VALIDATION SPLIT IS BY TRUCK, NOT RANDOM
    Trucks 104 and 111 account for the large majority of your fail examples.
    A random split would put images from the same truck (same shooting
    session, lighting, background, camera position) on both sides of the
    split - the model could then learn to recognize THAT TRUCK rather than
    genuine thermal-shock color patterns, and still score well on a random
    validation set without actually generalizing. Splitting by truck (via
    GroupKFold) prevents that shortcut: entire trucks are held out together.

    Caveat worth knowing: because fails are so concentrated in just 2 of
    ~22 trucks, some cross-validation folds may end up with very few or
    even zero true fails in their validation set, depending on which trucks
    land where. Per-fold fail counts are printed so you can see when a
    fold's numbers are less meaningful due to this.

Usage:
    python train_thermal_severity_regressor.py --manifest training_set/manifest.csv --epochs 30

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
from sklearn.metrics import (classification_report, confusion_matrix, mean_absolute_error,
                              mean_squared_error, r2_score)
from sklearn.model_selection import GroupKFold, KFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class SeverityDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_filepath"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        target = torch.tensor(row["dE_max"], dtype=torch.float32)
        return image, target


class NativeResSeverityDataset(Dataset):
    """Loads images at whatever resolution they're saved at - no resize.
    Use with pad_collate below. Mirrors NativeResPassFailDataset in
    train_pass_fail_classifier_native.py - keep the two in sync if you
    change one, since they're meant to consume the same preprocessed data."""

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
        target = torch.tensor(row["dE_max"], dtype=torch.float32)
        return tensor, target


PAD_VALUE = 0.5  # mid-gray in [0,1] range, applied before normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def pad_collate(batch):
    """Pads every image in the batch to that batch's own max H/W with
    neutral gray, then stacks. Needed because native-resolution crops may
    not be identical in size (though with a shared CROP_BOXES value across
    well_size, as you're currently using, this ends up being a no-op)."""
    images, targets = zip(*batch)
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded = []
    for img in images:
        c, h, w = img.shape
        canvas = torch.full((c, max_h, max_w), PAD_VALUE)
        top, left = (max_h - h) // 2, (max_w - w) // 2
        canvas[:, top:top + h, left:left + w] = img
        padded.append(canvas)

    return torch.stack(padded), torch.stack(targets)


def normalize_batch(batch_tensor, device):
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (batch_tensor.to(device) - mean) / std


def get_transforms(img_size=(256, 512), train=True):
    """img_size is (height, width) - since your preprocessed images are already
    resized non-square (matching your crop boxes' aspect ratio, no padding),
    this needs to match that shape rather than forcing a square."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            # Deliberately minimal color augmentation - the target IS a
            # color measurement, so aggressive brightness/contrast/color
            # jitter would corrupt the exact signal the model needs to learn,
            # the same reasoning applied to the mask segmentation script.
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def build_model(freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    return model


def predict_severity(model, x):
    """Softplus keeps predictions non-negative, since dE_max can't be negative,
    without hard-clamping (which would kill gradients at the boundary)."""
    return F.softplus(model(x)).squeeze(1)


def evaluate(model, loader, device, threshold, native=False):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = normalize_batch(images, device) if native else images.to(device)
            preds = predict_severity(model, images).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)

    pred_fail = (all_preds >= threshold).astype(int)
    true_fail = (all_targets >= threshold).astype(int)

    return {
        "mae": mae, "rmse": rmse, "r2": r2,
        "preds": all_preds, "targets": all_targets,
        "pred_fail": pred_fail, "true_fail": true_fail,
        "n_true_fail": int(true_fail.sum()), "n": len(all_targets),
    }


def train_one_split(train_df, val_df, device, img_size, epochs, lr, batch_size, threshold,
                     freeze_backbone=True, native=False):
    if native:
        train_ds = NativeResSeverityDataset(train_df, train=True)
        val_ds = NativeResSeverityDataset(val_df, train=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=2, collate_fn=pad_collate)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=2, collate_fn=pad_collate)
    else:
        train_ds = SeverityDataset(train_df, transform=get_transforms(img_size, train=True))
        val_ds = SeverityDataset(val_df, transform=get_transforms(img_size, train=False))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model(freeze_backbone).to(device)
    # Huber/SmoothL1 loss instead of plain MSE - less sensitive to the
    # occasional noisy dE_max value (remember dE_max comes from only 6
    # candidate points, not an exhaustive scan, so it's an imperfect proxy).
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)

    best_mae = float("inf")
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = normalize_batch(images, device) if native else images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = predict_severity(model, images)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_ds)

        val_metrics = evaluate(model, val_loader, device, threshold, native=native)
        scheduler.step(val_metrics["mae"])

        print(f"  Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | "
              f"val_MAE={val_metrics['mae']:.4f} val_RMSE={val_metrics['rmse']:.4f} "
              f"val_R2={val_metrics['r2']:.4f}")

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    final_metrics = evaluate(model, val_loader, device, threshold, native=native)
    return model, final_metrics


def print_downstream_classification(metrics, fold_label=""):
    print(f"\n  Downstream pass/fail classification{(' - ' + fold_label) if fold_label else ''} "
          f"(true fails in this val set: {metrics['n_true_fail']} of {metrics['n']}):")
    if metrics["n_true_fail"] == 0:
        print("  (No true fails in this validation set - classification metrics for the")
        print("   fail class aren't meaningful here; regression MAE/RMSE above still is.)")
        return
    print(classification_report(metrics["true_fail"], metrics["pred_fail"],
                                 target_names=["pass", "fail"], zero_division=0))
    print("  Confusion matrix (rows=true, cols=predicted), order: [pass, fail]")
    print(confusion_matrix(metrics["true_fail"], metrics["pred_fail"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to manifest.csv from build_thermal_training_set.py")
    parser.add_argument("--group_col", default="truck_number", help="Column to group by for the validation split")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--img_height", type=int, default=256,
                         help="Input height fed to the network - match this to what "
                              "preprocess_thermal_images.py actually produced "
                              "(--target_height there), not an arbitrary square size. "
                              "Your original crops are huge, and dE_max reflects color at "
                              "small specific worst-points, not a regional average - too "
                              "aggressive a downsample can blend those peaks away before the "
                              "network ever sees them. Compare CV R2 across resolutions "
                              "directly - don't assume, verify.")
    parser.add_argument("--img_width", type=int, default=512,
                         help="Input width fed to the network - match to --target_width used "
                              "in preprocessing")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=3.21,
                         help="dE_max threshold for pass/fail (default: your validated pooled threshold)")
    parser.add_argument("--unfreeze_backbone", action="store_true",
                         help="Fine-tune the whole network instead of just the final layer")
    parser.add_argument("--native", action="store_true",
                         help="Use images at native resolution (no resize) - matches manifests "
                              "produced by preprocess_thermal_images.py --no_resize. When set, "
                              "--img_height/--img_width are ignored; batching is handled via "
                              "per-batch padding instead of a fixed target size.")
    parser.add_argument("--output", default="thermal_severity_model.pt")
    parser.add_argument("--random_split_diagnostic", action="store_true",
                         help="Also run a plain random K-fold (NOT grouped by truck) purely as a "
                              "diagnostic - lets images from the same truck appear in both train "
                              "and val. This is NOT a valid generalization estimate (it's expected "
                              "to look better than the grouped CV for exactly that reason) - it "
                              "only answers one question: can the model learn the severity pattern "
                              "at all when given same-truck examples, or is it failing outright? "
                              "If this also comes back near/below R2=0, the problem isn't "
                              "generalization to new trucks - it's something more fundamental.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(args.manifest)
    print(f"Loaded {len(df)} rows from {args.manifest}")

    if args.group_col not in df.columns:
        print(f"'{args.group_col}' not found in manifest - can't do a group-aware split.")
        return

    n_missing_group = df[args.group_col].isna().sum()
    if n_missing_group > 0:
        print(f"\n{n_missing_group} row(s) have a missing '{args.group_col}' value - "
              f"excluding them, since they can't be assigned to a group-aware fold.")
        df = df[df[args.group_col].notna()].reset_index(drop=True)
        print(f"{len(df)} rows remain")

    n_missing_target = df["dE_max"].isna().sum()
    if n_missing_target > 0:
        print(f"{n_missing_target} row(s) have a missing 'dE_max' value - excluding them too.")
        df = df[df["dE_max"].notna()].reset_index(drop=True)
        print(f"{len(df)} rows remain")

    groups = df[args.group_col].values
    n_unique_groups = len(np.unique(groups))
    n_folds = min(args.n_folds, n_unique_groups)
    if n_folds < args.n_folds:
        print(f"Only {n_unique_groups} unique {args.group_col} values - reducing to {n_folds} folds")

    print(f"\n=== Cross-validation ({n_folds} folds, grouped by {args.group_col}) ===")
    gkf = GroupKFold(n_splits=n_folds)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        val_trucks = sorted(val_df[args.group_col].unique())

        print(f"\n--- Fold {fold + 1}/{n_folds} (val {args.group_col}s: {val_trucks}) ---")
        _, metrics = train_one_split(
            train_df, val_df, device, (args.img_height, args.img_width), args.epochs, args.lr,
            args.batch_size, args.threshold, freeze_backbone=not args.unfreeze_backbone,
            native=args.native,
        )
        print(f"Fold {fold + 1} best: MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f}")
        print_downstream_classification(metrics, fold_label=f"fold {fold + 1}")
        fold_metrics.append(metrics)

    print(f"\n=== Cross-validation summary ===")
    maes = [m["mae"] for m in fold_metrics]
    rmses = [m["rmse"] for m in fold_metrics]
    r2s = [m["r2"] for m in fold_metrics]
    print(f"MAE:  {np.mean(maes):.4f} (std {np.std(maes):.4f})")
    print(f"RMSE: {np.mean(rmses):.4f} (std {np.std(rmses):.4f})")
    print(f"R2:   {np.mean(r2s):.4f} (std {np.std(r2s):.4f})")

    # Pool predictions across all folds for one honest, full-dataset
    # downstream classification report (more reliable than any single
    # fold, especially the ones with few/zero true fails in validation).
    all_preds = np.concatenate([m["preds"] for m in fold_metrics])
    all_targets = np.concatenate([m["targets"] for m in fold_metrics])
    pooled_metrics = {
        "pred_fail": (all_preds >= args.threshold).astype(int),
        "true_fail": (all_targets >= args.threshold).astype(int),
        "n_true_fail": int((all_targets >= args.threshold).sum()),
        "n": len(all_targets),
    }
    print("\n=== Pooled downstream classification across all CV folds ===")
    print_downstream_classification(pooled_metrics)

    if args.random_split_diagnostic:
        print(f"\n{'='*70}")
        print(f"=== DIAGNOSTIC: plain random {n_folds}-fold (NOT grouped by truck) ===")
        print(f"{'='*70}")
        print("This is NOT a real evaluation - images from the same truck can land in")
        print("both train and val here, so it's expected to look better than the grouped")
        print("CV above even if the model has learned nothing generalizable. Its only")
        print("purpose: check whether the model CAN fit the severity pattern when given")
        print("same-truck examples on both sides, or whether it fails outright even then.")

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        random_fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            print(f"\n--- Random fold {fold + 1}/{n_folds} ---")
            _, metrics = train_one_split(
                train_df, val_df, device, (args.img_height, args.img_width), args.epochs, args.lr,
                args.batch_size, args.threshold, freeze_backbone=not args.unfreeze_backbone,
                native=args.native,
            )
            print(f"Random fold {fold + 1} best: MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} "
                  f"R2={metrics['r2']:.4f}")
            random_fold_metrics.append(metrics)

        r_maes = [m["mae"] for m in random_fold_metrics]
        r_rmses = [m["rmse"] for m in random_fold_metrics]
        r_r2s = [m["r2"] for m in random_fold_metrics]
        print(f"\n=== Random-split diagnostic summary ===")
        print(f"MAE:  {np.mean(r_maes):.4f} (std {np.std(r_maes):.4f})")
        print(f"RMSE: {np.mean(r_rmses):.4f} (std {np.std(r_rmses):.4f})")
        print(f"R2:   {np.mean(r_r2s):.4f} (std {np.std(r_r2s):.4f})")

        print(f"\nCompare to grouped CV R2: {np.mean(r2s):.4f}")
        if np.mean(r_r2s) > 0.3 and np.mean(r2s) < 0.1:
            print("-> Random split learns the pattern fine, grouped split doesn't: this points")
            print("   to a GENERALIZATION problem (not enough diverse severe examples across")
            print("   trucks), not a fundamental modeling failure. More severe examples from")
            print("   MORE distinct trucks would likely help more than further preprocessing.")
        elif np.mean(r_r2s) < 0.1:
            print("-> Even the random split struggles: the model isn't learning the pattern")
            print("   even with same-truck examples on both sides. This points to something")
            print("   more fundamental than generalization - worth revisiting the target itself,")
            print("   the crop/ROI content, or model capacity, before more preprocessing work.")
        else:
            print("-> Mixed result - review both sets of numbers together before concluding.")

    print(f"\n=== Training final deployment model on ALL {len(df)} labeled images ===")
    print("(Note: the val_MAE/RMSE/R2 printed during THIS run overlap with training data -")
    print(" they are not a validation metric. Use the CV summary above for the real estimate.)")
    final_model, _ = train_one_split(
        df, df.sample(min(30, len(df)), random_state=42),  # tiny internal check only
        device, (args.img_height, args.img_width), args.epochs, args.lr, args.batch_size, args.threshold,
        freeze_backbone=not args.unfreeze_backbone, native=args.native,
    )

    torch.save({
        "model_state_dict": final_model.state_dict(),
        "img_size": (args.img_height, args.img_width),
        "threshold": args.threshold,
    }, args.output)
    print(f"\nSaved final model to {args.output}")
    print(f"Expected real-world performance (from cross-validation): "
          f"MAE={np.mean(maes):.4f}, RMSE={np.mean(rmses):.4f}")


if __name__ == "__main__":
    main()
