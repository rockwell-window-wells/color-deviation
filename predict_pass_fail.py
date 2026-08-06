"""
Run inference with a trained pass/fail classifier.

Usage:
    python predict_pass_fail.py --model pass_fail_model.pt --image path/to/image.jpg
    python predict_pass_fail.py --model pass_fail_model.pt --folder path/to/images/
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, idx_to_class


def preprocess(image_path, img_size=224):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return tf(img).unsqueeze(0)


def predict(model, idx_to_class, image_path, device):
    x = preprocess(image_path).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

    return idx_to_class[pred_idx], probs[pred_idx].item(), {
        idx_to_class[i]: round(probs[i].item(), 4) for i in range(len(idx_to_class))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained .pt model file")
    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--folder", help="Path to a folder of images to classify")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx_to_class = load_model(args.model, device)

    if args.image:
        label, confidence, all_probs = predict(model, idx_to_class, args.image, device)
        print(f"{args.image}: {label} (confidence={confidence:.4f}) | probs={all_probs}")

    elif args.folder:
        files = sorted(f for f in Path(args.folder).iterdir() if f.suffix.lower() in IMG_EXTENSIONS)
        for f in files:
            label, confidence, all_probs = predict(model, idx_to_class, f, device)
            print(f"{f.name}: {label} (confidence={confidence:.4f}) | probs={all_probs}")

    else:
        print("Provide either --image or --folder")


if __name__ == "__main__":
    main()
