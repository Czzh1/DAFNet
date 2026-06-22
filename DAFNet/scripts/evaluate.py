#!/usr/bin/env python3
"""
Evaluation script for DAFNet.

Usage:
    python scripts/evaluate.py \\
        --checkpoint path/to/checkpoint.pth \\
        --dataset_path /path/to/dataset/test \\
        --model_name dafnet_tiny \\
        --num_classes 3
"""

import argparse
import os
import sys
import json

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score,
    cohen_kappa_score, matthews_corrcoef, confusion_matrix, roc_auc_score,
    classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dafnet.model import create_model
from dafnet.utils import read_dataset, MyDataSet
from dafnet.trainer import evaluate


def main():
    parser = argparse.ArgumentParser(
        description="DAFNet Model Evaluation"
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint (.pth)',
    )
    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='Path to test dataset folder',
    )
    parser.add_argument(
        '--model_name', type=str, default='dafnet_tiny',
        help='Model architecture name',
    )
    parser.add_argument(
        '--num_classes', type=int, default=3,
        help='Number of output classes',
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for evaluation',
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='Device to use',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to save results JSON',
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load class mapping
    class_indices_path = 'class_indices.json'
    if os.path.exists(class_indices_path):
        with open(class_indices_path, 'r') as f:
            idx_to_class = json.load(f)
    else:
        idx_to_class = None

    # Create model
    model = create_model(args.model_name, num_classes=args.num_classes)
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["net"], strict=False)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load data
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    images, labels = read_dataset(args.dataset_path)
    dataset = MyDataSet(
        images_path=images, images_class=labels, transform=transform,
    )
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=min(os.cpu_count(), 8),
        collate_fn=dataset.collate_fn,
    )

    # Evaluate
    results = evaluate(model, data_loader, device, epoch=None)

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for key, value in results.items():
        if key != 'cm':
            print(f"  {key}: {value}")

    print("\nConfusion Matrix:")
    cm = np.array(results['cm'])
    print(cm)

    if idx_to_class:
        print("\nClass mapping:")
        for idx, name in sorted(idx_to_class.items(), key=lambda x: int(x[0])):
            print(f"  {idx}: {name}")

    # Save results
    if args.output:
        results['cm'] = results['cm']  # already a list
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params / 1e6:.2f}M")


if __name__ == '__main__':
    main()