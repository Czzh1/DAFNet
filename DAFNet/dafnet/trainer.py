"""
Training and evaluation loop for DAFNet.

Provides:
- train_one_epoch: Single training epoch
- evaluate: Model evaluation with comprehensive metrics
- Trainer: Full training pipeline with cross-validation, early stopping, and logging
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score,
    cohen_kappa_score, matthews_corrcoef, confusion_matrix, roc_auc_score,
)
from tqdm import tqdm

from .utils import (
    my_print, read_dataset, MyDataSet,
    initialize_weights, get_params_groups, create_lr_scheduler, EarlyStopping,
)
from .model import create_model


def calculate_specificity(cm):
    """Calculate specificity from a confusion matrix."""
    n = cm.shape[0]
    specificities = []
    for i in range(n):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificities.append(spec)
    return np.mean(specificities)


def get_data_loader(images_path, labels, transform, batch_size, num_workers, shuffle=True):
    """Create a DataLoader from image paths and labels."""
    dataset = MyDataSet(
        images_path=images_path, images_class=labels, transform=transform
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        pin_memory=True, num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    """Train the model for one epoch.

    Returns:
        Tuple of (average_loss, accuracy, epoch_time_seconds)
    """
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = 0.0
    accu_num = 0
    sample_num = 0
    start_time = time.time()
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        sample_num += images.size(0)

        outputs = model(images)
        loss = loss_function(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        preds = torch.argmax(outputs, dim=1)
        accu_num += (preds == labels).sum().item()
        accu_loss += loss.item()

        data_loader.set_description(
            f"[train epoch {epoch}] loss: {accu_loss / (step + 1):.3f}, "
            f"acc: {accu_num / sample_num:.3f}, "
            f"lr: {optimizer.param_groups[0]['lr']:.5f}"
        )

    end_time = time.time()
    epoch_time = end_time - start_time
    return accu_loss / (step + 1), accu_num / sample_num, epoch_time


@torch.no_grad()
def evaluate(model, data_loader, device, epoch=None):
    """Evaluate the model on a dataset.

    Args:
        model: The model to evaluate
        data_loader: DataLoader for the evaluation set
        device: torch device
        epoch: Current epoch number (None for test set evaluation)

    Returns:
        Dict containing loss, accuracy, recall, f1, specificity, confusion_matrix,
        kappa, precision, mcc, auc, and inference_time
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    accu_loss, accu_num, sample_num = 0.0, 0, 0
    start_time = time.time()
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)

        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        accu_loss += loss.item()
        accu_num += (preds == labels).sum().item()
        sample_num += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        if epoch is None:
            prefix = "[Test ]"
        else:
            prefix = f"[Valid epoch {epoch}]"
        data_loader.set_description(
            f"{prefix} loss: {accu_loss / (step + 1):.3f}, "
            f"acc: {accu_num / sample_num:.3f}"
        )

    end_time = time.time()
    total_inference_time = end_time - start_time
    inference_time = total_inference_time / sample_num if sample_num > 0 else 0

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    specificity = calculate_specificity(cm)
    kappa = cohen_kappa_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)

    all_labels_one_hot = F.one_hot(
        torch.tensor(all_labels), num_classes=probs.size(1)
    ).numpy()
    auc = roc_auc_score(
        all_labels_one_hot, all_probs, average='macro', multi_class='ovr'
    )

    return {
        "loss": round(accu_loss / (step + 1), 3),
        "accuracy": round(accuracy, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "specificity": round(specificity, 3),
        "cm": cm.tolist(),
        "kappa": round(kappa, 3),
        "precision": round(precision, 3),
        "mcc": round(mcc, 3),
        "auc": round(auc, 3),
        "inference_time": round(inference_time, 6),
    }


class Trainer:
    """Full training pipeline with cross-validation.

    Args:
        args: Dictionary of training arguments (see train.py for keys)
        logger: Logger instance
    """

    def __init__(self, args: dict, logger=None):
        self.args = args
        self.logger = logger
        self.device = torch.device(
            f'cuda:{args["device"].split(",")[0]}'
            if torch.cuda.is_available() else "cpu"
        )
        self.img_size = 224
        self._setup_transforms()

    def _setup_transforms(self):
        """Set up data transforms for train/val/test."""
        self.data_transform = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]),
            "val": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]),
            "test": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]),
        }

    def run(self) -> dict:
        """Run the full training pipeline with cross-validation.

        Returns:
            Dict of aggregated metrics (mean ± std across folds)
        """
        args = self.args
        device = self.device

        # Load data
        train_val_images, train_val_labels = read_dataset(
            os.path.join(args['dataset_path'], 'train_val')
        )
        test_images, test_labels = read_dataset(
            os.path.join(args['dataset_path'], 'test')
        )

        test_loader = get_data_loader(
            test_images, test_labels,
            self.data_transform["test"],
            args['batch_size'], min(os.cpu_count(), 8),
        )

        cv_folds = args["cv_folds"]
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=44)

        # Metrics accumulator
        all_metrics = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'val_f1': [], 'val_recall': [], 'val_precision': [], 'val_mcc': [],
            'val_kappa': [], 'val_auc': [], 'val_specificity': [],
            'epoch_time': [], 'inference_time': [],
            'test_acc': [], 'test_recall': [], 'test_f1': [], 'test_precision': [],
            'test_mcc': [], 'test_kappa': [], 'test_auc': [], 'test_specificity': [],
            'flops': [], 'num_parameters': [],
        }

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_val_images, train_val_labels)
        ):
            train_loader = get_data_loader(
                np.array(train_val_images)[train_idx],
                np.array(train_val_labels)[train_idx],
                self.data_transform["train"],
                args['batch_size'], min(os.cpu_count(), 8),
            )
            val_loader = get_data_loader(
                np.array(train_val_images)[val_idx],
                np.array(train_val_labels)[val_idx],
                self.data_transform["val"],
                args['batch_size'], min(os.cpu_count(), 8),
            )

            # Create model
            model = create_model(
                args['model_name'], num_classes=args["num_classes"]
            ).to(device)
            initialize_weights(model)

            # Compute FLOPs (requires thop)
            try:
                from thop import profile
                dummy_input = torch.randn(1, 3, self.img_size, self.img_size).to(device)
                flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            except ImportError:
                flops = 0

            my_print(
                f"Starting Training for Fold {fold + 1}/{cv_folds} "
                f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", self.logger,
            )
            my_print(args, self.logger)
            my_print(
                f'Using {args["num_workers"]} dataloader workers, '
                f'FLOPs: {flops}', self.logger,
            )
            all_metrics['flops'].append(flops)

            num_params = sum(p.numel() for p in model.parameters())
            all_metrics['num_parameters'].append(num_params)

            # Optimizer and scheduler
            optimizer = optim.AdamW(
                get_params_groups(model, weight_decay=args["weight_decay"]),
                lr=args["learning_rate"],
            )
            lr_scheduler = create_lr_scheduler(
                optimizer, len(train_loader), args["epochs"], warmup=True,
            )

            early_stopping = EarlyStopping(
                patience=30, monitor='val_loss', delta=0.001, logger=self.logger,
            )

            best_weight_file = ""
            best_accuracy = 0.0
            early_bool = False

            # Training loop
            for epoch in range(1, args["epochs"] + 1):
                train_loss, train_acc, epoch_time = train_one_epoch(
                    model, optimizer, train_loader, device, epoch, lr_scheduler,
                )
                val_results = evaluate(model, val_loader, device, epoch=epoch)

                my_print(
                    f"epoch: {epoch}[train_loss: {train_loss:.3f}, "
                    f"train_acc: {train_acc:.3f}, epoch_time: {epoch_time:.3f}s] "
                    f"\n[val_results: {val_results}]", self.logger,
                )

                # Record metrics
                all_metrics['train_loss'].append(train_loss)
                all_metrics['train_acc'].append(train_acc)
                all_metrics['val_loss'].append(val_results["loss"])
                all_metrics['val_acc'].append(val_results["accuracy"])
                all_metrics['val_f1'].append(val_results["f1"])
                all_metrics['val_recall'].append(val_results["recall"])
                all_metrics['val_precision'].append(val_results["precision"])
                all_metrics['val_mcc'].append(val_results["mcc"])
                all_metrics['val_kappa'].append(val_results["kappa"])
                all_metrics['val_auc'].append(val_results["auc"])
                all_metrics['val_specificity'].append(val_results["specificity"])
                all_metrics['epoch_time'].append(epoch_time)
                all_metrics['inference_time'].append(val_results["inference_time"])

                # TensorBoard
                tb_writer = SummaryWriter()
                tb_writer.add_scalar("train_loss", train_loss, epoch)
                tb_writer.add_scalar("train_acc", train_acc, epoch)
                tb_writer.add_scalar("val_loss", val_results["loss"], epoch)
                tb_writer.add_scalar("val_acc", val_results["accuracy"], epoch)
                tb_writer.add_scalar("val_f1", val_results["f1"], epoch)
                tb_writer.add_scalar("val_auc", val_results["auc"], epoch)
                tb_writer.add_scalar("epoch_time", epoch_time, epoch)
                tb_writer.add_scalar(
                    "inference_time", val_results["inference_time"], epoch,
                )

                # Early stopping
                early_stopping(val_results["loss"], model, epoch)
                if early_stopping.early_stop:
                    my_print(f"Early stopping at epoch: {epoch}...", self.logger)
                    early_bool = True

                # Save checkpoints
                model_weight_path = os.path.join(
                    args["model_save_path"], args['model_name'],
                    f"fold_{fold + 1}",
                )
                os.makedirs(model_weight_path, exist_ok=True)

                if val_results["accuracy"] > best_accuracy:
                    best_accuracy = val_results["accuracy"]
                    weight_file = os.path.join(
                        model_weight_path,
                        f"best_ckpt_epoch_{epoch}_acc_{val_results['accuracy']:.3f}.pth",
                    )
                    checkpoint = {
                        "net": model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        "epoch": epoch,
                        'lr_schedule': lr_scheduler.state_dict(),
                    }
                    if best_weight_file:
                        try:
                            os.remove(best_weight_file)
                            my_print(
                                f"Removed old best model: {best_weight_file}",
                                self.logger,
                            )
                        except FileNotFoundError:
                            pass
                    torch.save(checkpoint, weight_file)
                    best_weight_file = weight_file
                    my_print(
                        f"Saved epoch={epoch}, accuracy={val_results['accuracy']} "
                        f"as new best model.....", self.logger,
                    )

                if epoch == args["epochs"] or early_bool:
                    checkpoint = {
                        "net": model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        "epoch": epoch,
                        'lr_schedule': lr_scheduler.state_dict(),
                    }
                    torch.save(
                        checkpoint,
                        f"{model_weight_path}/last_ckpt_epoch_{epoch}_"
                        f"acc_{val_results['accuracy']:.3f}.pth",
                    )

                my_print(
                    f'epoch: {epoch} learning rate: '
                    f'{optimizer.state_dict()["param_groups"][0]["lr"]}\n',
                    self.logger,
                )

                if early_bool:
                    break

            # Test set evaluation
            checkpoint = torch.load(best_weight_file, map_location=device)
            model.load_state_dict(checkpoint["net"])
            test_results = evaluate(model, test_loader, device, epoch=None)
            my_print(f"test_results: {test_results}\n", self.logger)

            all_metrics['test_acc'].append(test_results["accuracy"])
            all_metrics['test_recall'].append(test_results["recall"])
            all_metrics['test_f1'].append(test_results["f1"])
            all_metrics['test_precision'].append(test_results["precision"])
            all_metrics['test_mcc'].append(test_results["mcc"])
            all_metrics['test_kappa'].append(test_results["kappa"])
            all_metrics['test_auc'].append(test_results["auc"])
            all_metrics['test_specificity'].append(test_results["specificity"])

        # Aggregate results
        result_metrics = {}
        for metric in [
            'test_acc', 'test_recall', 'test_f1', 'test_precision',
            'test_mcc', 'test_kappa', 'test_auc', 'test_specificity',
        ]:
            if all_metrics[metric]:
                result_metrics[metric] = (
                    f"{np.mean(all_metrics[metric]):.3f}+/-"
                    f"{np.std(all_metrics[metric]):.3f}"
                )

        result_metrics['epoch_time'] = (
            f"{np.mean(all_metrics['epoch_time']):.3f}+/-"
            f"{np.std(all_metrics['epoch_time']):.3f}s"
        )
        result_metrics['inference_time'] = (
            f"{np.mean(all_metrics['inference_time']) * 1000:.3f}+/-"
            f"{np.std(all_metrics['inference_time']) * 1000:.3f}ms"
        )
        result_metrics["num_parameters"] = (
            f"{np.mean(all_metrics['num_parameters']) / 1e6:.3f}M"
        )
        result_metrics["flops"] = f"{int(np.mean(all_metrics['flops']))}"

        my_print("Final Test Metrics:", self.logger)
        for metric in result_metrics:
            my_print(f"{metric}: {result_metrics[metric]}", self.logger)

        return result_metrics