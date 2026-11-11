"""
Exercise 02 - Tensors and Tensor Operations
============================================
Solve each task below. Run this script to verify your solutions:
    python tensors.py
"""

import torch


# ===========================================================================
# Task 1: Tensor Creation and Inspection
# ===========================================================================
# Create a random tensor of shape (2, 3, 224, 224) — simulating a batch
# of 2 RGB images of size 224x224.
# Print its shape, dtype, and device.
#
# Hint: torch.randn() creates a tensor with random values from a
# normal distribution. Pass the dimensions as arguments.


images = torch.randn(2, 3, 224, 224)
print("Task 1:", images.shape, images.dtype, images.device)

# ===========================================================================
# Task 2: Device Transfer
# ===========================================================================
# Move the tensor from Task 1 to GPU (if available).
# Print the device before and after the transfer.
#
# Step 1: Check if CUDA is available using torch.cuda.is_available().
# Step 2: Move the tensor using .to('cuda').
# Step 3: Print the .device attribute to confirm.


print("Device before:", images.device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
images = images.to(device)
print("Device after:", images.device)

# ===========================================================================
# Task 3: Indexing and Slicing
# ===========================================================================
# Create a random tensor of shape (8, 3, 32, 32) — a batch of 8 images.
#
# Step 1: Extract the 5th image from the batch. Remember that Python
#         indexing starts at 0, so the 5th image is at index 4.
#         Print its shape — should be (3, 32, 32).
#
# Step 2: Extract only the green channel (index 1) of that image.
#         Print its shape — should be (32, 32).


batch = torch.randn(8, 3, 32, 32)
fifth_image = batch[4]
green_channel = fifth_image[1]
print("Task 3:", fifth_image.shape, green_channel.shape)

# ===========================================================================
# Task 4: Permute — HWC to CHW
# ===========================================================================
# Create a random tensor of shape (64, 64, 3) representing an image
# in HWC format (Height, Width, Channels).
#
# Step 1: Convert to PyTorch's CHW format using .permute().
#         You need to reorder dimensions: dim 2 → 0, dim 0 → 1, dim 1 → 2.
# Step 2: Print both shapes to confirm (64, 64, 3) → (3, 64, 64).


image_hwc = torch.randn(64, 64, 3)
image_chw = image_hwc.permute(2, 0, 1)
print("Task 4:", image_hwc.shape, image_chw.shape)

# ===========================================================================
# Task 5: Squeeze and Unsqueeze
# ===========================================================================
# a) Create a tensor of shape (3, 128, 128) representing a single image.
#    Add a batch dimension at position 0 using .unsqueeze().
#    Print the shape — should be (1, 3, 128, 128).
#
# b) Remove the batch dimension using .squeeze() and verify you get
#    back to (3, 128, 128).
#
# This pattern is common: models expect batched input [B, C, H, W],
# so a single image needs unsqueeze before passing through.


image = torch.randn(3, 128, 128)
batched_image = image.unsqueeze(0)
restored_image = batched_image.squeeze(0)
print("Task 5:", batched_image.shape, restored_image.shape)

# ===========================================================================
# Task 6: Reshaping — Flatten
# ===========================================================================
# Create a random tensor of shape (4, 16, 8, 8) — simulating a batch
# of 4 feature maps with 16 channels and 8x8 spatial dimensions.
#
# Flatten each sample so the result has shape (4, 1024).
# Note: 16 * 8 * 8 = 1024.
#
# Use .view() or .reshape(). The batch dimension (4) stays, everything
# else gets flattened into one dimension. You can pass -1 to let
# PyTorch infer the flattened size automatically.


feature_maps = torch.randn(4, 16, 8, 8)
flattened = feature_maps.reshape(feature_maps.size(0), -1)
print("Task 6:", flattened.shape)

# ===========================================================================
# Task 7: Basic Math
# ===========================================================================
# a) Create two random tensors of shape (3, 3).
#    Compute their element-wise product (use * operator) and their
#    matrix product (use @ operator). Print both results.
#
# b) Create a random tensor of shape (4, 3, 32, 32).
#    Compute the mean pixel value across the batch dimension (dim=0).
#    Print the resulting shape — should be (3, 32, 32).
#    This is how dataset mean is computed for normalization.

a = torch.randn(3, 3)
b = torch.randn(3, 3)
print("Element-wise product:\n", a * b)
print("Matrix product:\n", a @ b)
pixel_batch = torch.randn(4, 3, 32, 32)
mean_image = pixel_batch.mean(dim=0)
print("Batch mean shape:", mean_image.shape)
