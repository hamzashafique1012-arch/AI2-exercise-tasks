"""
Exercise 03 - Training Script
================================
Main training script that ties together dataset, model, loss, optimizer,
TensorBoard logging, and checkpointing.

Usage:
    python src/train.py --config config.yaml

Project structure inspired by: https://github.com/victoresque/pytorch-template
"""

import os
import sys
import argparse

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import PlantWildDataset
from transforms import get_train_transforms, get_val_transforms
from model import PlantCNN, count_parameters
from utils import set_seed, compute_accuracy, save_checkpoint


def load_config(config_path):
    """Load YAML config file and return as dictionary."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_one_epoch(model, dataloader, criterion, optimizer, device, log_interval=10):
    """
    Train the model for one epoch.

    Returns:
        tuple: (average_loss, average_accuracy) for the epoch.
    """
    # Task: Set the model to training mode (.train).
    #       This enables dropout and batch normalization updates.


    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        # Task: Complete the training step.
        #
        #   Step 1: Move images and labels to the device using .to(device).
        #
        #   Step 2: Clear old gradients from the optimizer.
        #
        #   Step 3: Forward pass — get model outputs from the images.
        #
        #   Step 4: Compute the loss using the criterion.
        #           Pass (outputs, labels) — not (labels, outputs).
        #
        #   Step 5: Backward pass — compute gradients from the loss.
        #
        #   Step 6: Update weights with the optimizer.


        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # --- Accumulate metrics (do not modify) ---
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

        if (batch_idx + 1) % log_interval == 0:
            print(f"    Batch [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """
    Evaluate the model on the validation set.

    Returns:
        tuple: (average_loss, average_accuracy) for the validation set.
    """
    model.eval()
    # Set the model to evaluation mode (.eval).
    #       This disables dropout and freezes batch normalization.


    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    # Task: Disable gradient computation for the entire validation loop.
    #       Wrap the loop in a torch.no_grad() context manager.
    #       Inside, for each batch:
    #
    #   Step 1: Move images and labels to device.
    #   Step 2: Forward pass only — no backward, no optimizer step.
    #   Step 3: Compute loss (for logging, not for training).
    #
    #   The metric accumulation code below handles the rest.

    for images, labels in dataloader:

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # --- Accumulate metrics (do not modify) ---
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def main(config_path):
    """Main training function."""

    config = load_config(config_path)
    print(f"Loaded config from: {config_path}")

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Task: Create the TensorBoard SummaryWriter. (Hint: https://docs.pytorch.org/docs/2.14/tensorboard.html)
    #   The log directory should include the experiment name so each
    #   experiment gets its own subfolder. Combine config["logging"]["log_dir"]
    #   and config["experiment_name"] using os.path.join().
    log_dir = os.path.join(config["logging"]["log_dir"], config["experiment_name"])
    writer = SummaryWriter(log_dir=log_dir)

    image_size = config["data"]["image_size"]
    augmentation = config["data"]["augmentation"]

    # Task: Create training and validation datasets.
    #   Use PlantWildDataset with the appropriate root_dir
    #   (os.path.join with "train" or "val") and transforms.
    # Check the config file for specific keywords related to root directory and datasets
    train_dataset = PlantWildDataset(
        os.path.join(config["data"]["root_dir"], "train"),
        transform=get_train_transforms(image_size, augmentation),
    )
    val_dataset = PlantWildDataset(
        os.path.join(config["data"]["root_dir"], "val"),
        transform=get_val_transforms(image_size),
    )
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("Training and validation datasets must be non-empty")
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("Training and validation class mappings must match")
    if config["model"]["num_classes"] != len(train_dataset.classes):
        raise ValueError("model.num_classes must match the dataset class count")

    # Task: Create DataLoaders for both datasets.
    #   Use batch_size and num_workers from config["data"].
    #   Training: shuffle=True. Validation: shuffle=False.
    train_loader = DataLoader(
        train_dataset, batch_size=config["data"]["batch_size"],
        shuffle=True, num_workers=config["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["data"]["batch_size"],
        shuffle=False, num_workers=config["data"]["num_workers"],
    )

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # Task: Instantiate the model.
    #   Use PlantCNN with num_classes and dropout from config.
    #   Move to device.
    model = PlantCNN(num_classes=config["model"]["num_classes"],
                     dropout=config["model"]["dropout"]).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Task: Define the loss function — nn.CrossEntropyLoss().
    criterion = nn.CrossEntropyLoss()

    # Task: Define the optimizer based on config["training"]["optimizer"].
    #   If "adam", use torch.optim.Adam.
    #   If "sgd", use torch.optim.SGD with momentum=0.9.
    #   Pass learning_rate and weight_decay from config.
    optimizer_name = config["training"]["optimizer"].lower()
    optimizer_args = dict(lr=config["training"]["learning_rate"],
                          weight_decay=config["training"]["weight_decay"])
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_args)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, **optimizer_args)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Task: Log sample images to TensorBoard for visual verification.
    #   Fetch one batch from train_loader using next(iter(...)).
    #   Log images using writer.add_images("sample_training_images", images, 0).


    images, _ = next(iter(train_loader))
    # Undo ImageNet normalization so TensorBoard displays the true colors.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    writer.add_images("sample_training_images", (images * std + mean).clamp(0, 1), 0)

    # --- Training Loop ---
    best_val_acc = -1.0
    num_epochs = config["training"]["epochs"]

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            log_interval=config["logging"]["log_interval"],
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Task: Log four scalar metrics to TensorBoard using writer.add_scalar():
        #   "Loss/train", "Loss/val", "Accuracy/train", "Accuracy/val"
        #   Use epoch as the global_step.


        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.flush()

        # Save checkpoint if validation accuracy improved.
        #   Compare val_acc with best_val_acc. If better, update best_val_acc
        #   and call save_checkpoint(). Save to the path built from
        #   config["checkpoint"]["save_dir"] and experiment_name + "_best.pt".


        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, os.path.join(
                config["checkpoint"]["save_dir"], config["experiment_name"] + "_best.pt"))
        if not config["checkpoint"].get("save_best_only", True):
            save_checkpoint(model, optimizer, epoch, val_acc, os.path.join(
                config["checkpoint"]["save_dir"],
                config["experiment_name"] + f"_epoch_{epoch:03d}.pt"))

    writer.close()

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PlantCNN")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    args = parser.parse_args()

    main(args.config)
