# YOLO Instance Segmentation Format - Complete Explanation

This document explains the YOLO instance segmentation format based on comprehensive analysis and investigation.

## Overview

YOLO instance segmentation format provides pixel-level masks for object instances while maintaining the relationship between multiple objects of the same class within a single image.

## Data Structure Components

### 1. **`masks`** - List of 2D Tensors
- **Type**: `List[torch.Tensor]`
- **Shape**: Each tensor is `[H, W]` (height × width)
- **Count**: One mask per image in the batch
- **Purpose**: Contains pixel-level instance IDs for each image

### 2. **`cls`** - Class Labels Tensor
- **Type**: `torch.Tensor`
- **Shape**: `[N]` where N = total instances across all images
- **Content**: Class IDs for each instance (e.g., 22, 45, 50, 23, ...)
- **Purpose**: Maps each instance to its class label

### 3. **`batch_idx`** - Batch Index Tensor
- **Type**: `torch.Tensor` 
- **Shape**: `[N]` where N = total instances across all images
- **Content**: Image index for each instance (e.g., 0, 1, 1, 1, ...)
- **Purpose**: Indicates which image each instance belongs to

## Mask Encoding System

### Pixel Values in Masks:
- **Value 0**: Background pixels (empty space)
- **Value 1**: First instance in the image
- **Value 2**: Second instance in the image
- **Value 3**: Third instance in the image
- **...and so on**

### Key Properties:
1. **Background**: Always encoded as 0
2. **Instance IDs**: Start from 1 and are consecutive
3. **Per-image basis**: Instance numbering resets for each image
4. **Spatial separation**: Different instances have different pixel values

## Relationships and Mapping

### Core Relationships:
```
Number of masks = Number of images in batch
Number of cls entries = Total instances across all images
Number of batch_idx entries = Total instances across all images
```

### Instance-to-Class Mapping:
For each image `i`, to find the class of instance with pixel value `instance_id`:
```python
# Get all instances for image i
image_instances = cls[batch_idx == i]

# Map instance ID to class
class_label = image_instances[instance_id - 1]  # -1 because instance IDs start from 1
```

## Concrete Example

Consider a batch with 2 images:

### Batch Data:
```python
cls = [22, 45, 50, 45, 45, 23, 23, 23, 49, 23, 49, 49, 49]
batch_idx = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
masks = [mask_0, mask_1]  # Two 640x640 tensors
```

### Image 0 Analysis:
- **Instances**: 1 instance (class 22)
- **Mask values**: `[0, 1]` where:
  - `0` = background
  - `1` = instance with class 22

### Image 1 Analysis:
- **Instances**: 12 instances (classes: 45, 50, 45, 45, 23, 23, 23, 49, 23, 49, 49, 49)
- **Mask values**: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]` where:
  - `0` = background
  - `1` = first instance (class 45)
  - `2` = second instance (class 50)
  - `3` = third instance (class 45)
  - ...and so on

---

*This analysis was conducted using COCO8-seg dataset with comprehensive pixel-level investigation and statistical analysis.*
