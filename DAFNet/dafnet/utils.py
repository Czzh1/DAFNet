"""
Utility functions for DAFNet training and evaluation.

Includes:
- Dataset loading and MyDataSet class
- Training utilities (weight init, LR scheduler, early stopping, optimizer param groups)
- Visualization functions (confusion matrix, ROC curve, PR curve)
"""

import os
import json
import pickle
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MyDataSet(Dataset):
    """Custom dataset for image classification from folder-based organization.

    Expects data organized as:
        root/class_0/*.jpg
        root/class_1/*.jpg
        ...

    Args:
        images_path: List of image file paths
        images_class: List of integer class labels
        transform: torchvision transforms to apply
    """

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            img = img.convert("RGB")
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

    def filter_by_indices(self, indices):
        """Return a new dataset containing only the specified indices."""
        filtered_images_path = [self.images_path[i] for i in indices]
        filtered_images_class = [self.images_class[i] for i in indices]
        return MyDataSet(
            filtered_images_path, filtered_images_class, transform=self.transform
        )


def read_dataset(root: str):
    """Read a folder-based image dataset and return paths and labels.

    Args:
        root: Path to dataset root directory (one subfolder per class)

    Returns:
        Tuple of (image_paths, labels)
    """
    random.seed(44)
    assert os.path.exists(root), f"dataset root: {root} does not exist."

    category = [cls for cls in os.listdir(root)
                if os.path.isdir(os.path.join(root, cls))]
    category.sort()
    class_indices = dict((k, v) for v, k in enumerate(category))

    # Save class index mapping
    json_str = json.dumps(
        dict((val, key) for key, val in class_indices.items()), indent=4
    )
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    images_paths = []
    images_labels = []
    supported = [".jpg", ".JPG", ".png", ".PNG", ".tif", ".TIF", ".jpeg", ".JPEG"]

    for cls in category:
        cls_path = os.path.join(root, cls)
        images = [
            os.path.join(root, cls, i)
            for i in os.listdir(cls_path)
            if os.path.splitext(i)[-1] in supported
        ]
        image_class = class_indices[cls]
        for img_path in images:
            images_paths.append(img_path)
            images_labels.append(image_class)

    print(f"{len(images_paths)} images found in {root}.")
    return images_paths, images_labels


def read_dataset_with_split(root: str, test_rate: float = 0.2):
    """Read dataset and split into train_val and test sets.

    Args:
        root: Path to dataset root directory
        test_rate: Fraction of data to use as test set

    Returns:
        Tuple of (train_val_paths, train_val_labels, test_paths, test_labels)
    """
    random.seed(44)
    assert os.path.exists(root), f"dataset root: {root} does not exist."

    num_classes = [cla for cla in os.listdir(root)
                   if os.path.isdir(os.path.join(root, cla))]
    num_classes.sort()
    class_indices = dict((k, v) for v, k in enumerate(num_classes))
    json_str = json.dumps(
        dict((val, key) for key, val in class_indices.items()), indent=4
    )
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_val_images, train_val_labels = [], []
    test_images, test_labels = [], []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG", ".tif", ".TIF", ".jpeg", ".JPEG"]

    for cla in num_classes:
        cla_path = os.path.join(root, cla)
        images = [
            os.path.join(root, cla, i)
            for i in os.listdir(cla_path)
            if os.path.splitext(i)[-1] in supported
        ]
        images.sort()
        image_class = class_indices[cla]
        every_class_num.append(len(images))

        val_path = random.sample(images, k=int(len(images) * test_rate))
        for img_path in images:
            if img_path in val_path:
                test_images.append(img_path)
                test_labels.append(image_class)
            else:
                train_val_images.append(img_path)
                train_val_labels.append(image_class)

    print(f"{sum(every_class_num)} images were found in the dataset.")
    print(f"{len(train_val_images)} images for training.")
    print(f"{len(test_images)} images for validation.")
    assert len(train_val_images) > 0, "number of training images must be > 0."
    assert len(test_images) > 0, "number of validation images must be > 0."

    return train_val_images, train_val_labels, test_images, test_labels


# ---------------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------------

def initialize_weights(model: nn.Module):
    """Initialize model weights using Kaiming normal initialization."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Optimizer parameter groups
# ---------------------------------------------------------------------------

def get_params_groups(model: nn.Module, weight_decay: float = 1e-5):
    """Split model parameters into weight-decay and no-weight-decay groups.

    Bias terms and 1D parameters (e.g., LayerNorm weights) get no weight decay.
    """
    parameter_group_vars = {
        "decay": {"params": [], "weight_decay": weight_decay},
        "no_decay": {"params": [], "weight_decay": 0.},
    }
    parameter_group_names = {
        "decay": {"params": [], "weight_decay": weight_decay},
        "no_decay": {"params": [], "weight_decay": 0.},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    return list(parameter_group_vars.values())


# ---------------------------------------------------------------------------
# Learning rate scheduler
# ---------------------------------------------------------------------------

def create_lr_scheduler(
    optimizer,
    num_step: int,
    epochs: int,
    warmup: bool = True,
    warmup_epochs: int = 1,
    warmup_factor: float = 1e-3,
    end_factor: float = 1e-2,
):
    """Create a cosine annealing LR scheduler with optional linear warmup.

    Args:
        optimizer: PyTorch optimizer
        num_step: Number of steps per epoch
        epochs: Total number of epochs
        warmup: Whether to use linear warmup
        warmup_epochs: Number of warmup epochs
        warmup_factor: Starting LR multiplier during warmup
        end_factor: Minimum LR multiplier (relative to base LR)

    Returns:
        LambdaLR scheduler
    """
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return (
                (1 + math.cos(current_step * math.pi / cosine_steps)) / 2
            ) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Early stopping handler.

    Args:
        patience: Number of epochs to wait before stopping
        delta: Minimum change to qualify as an improvement
        monitor: Metric to monitor ('val_loss' or 'val_acc')
        logger: Logger instance for recording messages
    """

    def __init__(self, patience=5, delta=0., monitor='val_loss', logger=None):
        self.patience = patience
        self.delta = delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.logger = logger

    def __call__(self, score, model, epoch):
        if self.best_score is None:
            self.best_score = score
        elif (
            (self.monitor == 'val_loss' and score > self.best_score + self.delta)
            or (self.monitor == 'val_acc' and score < self.best_score - self.delta)
        ):
            self.counter += 1
            my_print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}',
                self.logger,
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def my_print(msg, logger):
    """Print message to stdout and optionally to a logger."""
    print(msg)
    if logger is not None:
        logger.info(msg)


# ---------------------------------------------------------------------------
# Pickle helpers
# ---------------------------------------------------------------------------

def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true, y_pred, classes,
    normalize=False,
    title=None,
    cmap=plt.cm.Blues,
):
    """Plot a confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: List of class names
        normalize: Whether to normalize the matrix
        title: Plot title
        cmap: Colormap
    """
    if not title:
        title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes, yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label',
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return ax


def plot_roc_curve(y_true, y_scores, class_name=None):
    """Plot ROC curve and compute AUROC.

    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities for the positive class
        class_name: Optional class name for legend
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auroc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(5, 5))
    plt.plot(
        fpr, tpr,
        label=f'{class_name} ROC (area = {auroc:.2f})' if class_name
        else f'ROC (area = {auroc:.2f})',
    )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if class_name:
        plt.legend(loc="best")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.show()


def plot_precision_recall_curve(y_true, y_scores, class_name=None):
    """Plot Precision-Recall curve and compute AUPR.

    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities for the positive class
        class_name: Optional class name for legend
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    aupr = auc(recall, precision)

    plt.figure(figsize=(5, 5))
    plt.plot(
        recall, precision,
        label=f'{class_name} AUPR (area = {aupr:.2f})' if class_name
        else f'AUPR (area = {aupr:.2f})',
    )
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if class_name:
        plt.legend(loc="best")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.show()