#!/usr/bin/env python3
"""
Training entry point for DAFNet.

Usage:
    python scripts/train.py --dataset_path /path/to/dataset --model_name dafnet_tiny

The dataset should be organized as:
    dataset_path/
        train_val/
            class_0/
                img1.jpg
                ...
            class_1/
                img2.jpg
                ...
        test/
            class_0/
                img1.jpg
                ...
            class_1/
                img2.jpg
                ...
"""

import argparse
import csv
import json
import logging
import os
import sys

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dafnet.trainer import Trainer
from dafnet.model import list_models


def main():
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(
        description="DAFNet: Dynamic Attention Fusion Network - Training"
    )

    # Model
    parser.add_argument(
        '--model_name', type=str, default='dafnet_tiny',
        choices=list_models(),
        help='Model architecture to use',
    )
    parser.add_argument(
        '--num_classes', type=int, default=3,
        help='Number of output classes',
    )

    # Data
    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='Path to dataset root (contains train_val/ and test/ subdirs)',
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for training',
    )
    parser.add_argument(
        '--num_workers', type=int, default=8,
        help='Number of DataLoader workers',
    )

    # Training
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--cv_folds', type=int, default=5,
        help='Number of cross-validation folds',
    )
    parser.add_argument(
        '--learning_rate', type=float, default=0.001,
        help='Initial learning rate',
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.01,
        help='Weight decay for AdamW',
    )

    # System
    parser.add_argument(
        '--device', type=str, default='0',
        help='CUDA device index (e.g., "0" or "0,1")',
    )
    parser.add_argument(
        '--model_save_path', type=str, default='./outputs',
        help='Directory to save model checkpoints and logs',
    )
    parser.add_argument(
        '--resume', type=bool, default=False,
        help='Resume from checkpoint',
    )
    parser.add_argument(
        '--weights', type=str, default='',
        help='Path to pretrained weights',
    )
    parser.add_argument(
        '--freeze_layers', type=bool, default=False,
        help='Freeze backbone layers',
    )

    args = parser.parse_args()
    opt = vars(args)

    # Setup CSV logging
    os.makedirs(args.model_save_path, exist_ok=True)
    dataset_name = os.path.basename(args.dataset_path.rstrip('/'))
    csv_file_path = os.path.join(
        args.model_save_path, f"{dataset_name}_metrics.csv"
    )

    field_names = [
        "model_name", 'test_acc', 'test_recall', 'test_f1', 'test_precision',
        'test_mcc', 'test_kappa', 'test_auc', 'test_specificity',
        'epoch_time', 'inference_time', 'flops', 'num_parameters',
    ]

    if not os.path.exists(csv_file_path) or os.path.getsize(csv_file_path) == 0:
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()

    # Setup logger
    log_path = os.path.join(args.model_save_path, "log")
    os.makedirs(log_path, exist_ok=True)

    log_file_name = f"{args.model_name}_{dataset_name}.txt"
    log_file_path = os.path.join(log_path, log_file_name)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Train
    trainer = Trainer(opt, logger=logger)
    metrics = trainer.run()

    # Save to CSV
    row_data = {field_names[0]: args.model_name}
    for metric in field_names[1:]:
        row_data[metric] = metrics.get(metric, "N/A")

    print(row_data)
    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writerow(row_data)


if __name__ == '__main__':
    main()