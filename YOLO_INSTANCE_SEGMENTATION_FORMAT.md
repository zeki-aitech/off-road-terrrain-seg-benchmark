# YOLO Instance Segmentation Format

## Overview

YOLO (You Only Look Once) uses a specific data format for instance segmentation that differs from semantic segmentation. This document explains the standard YOLO instance segmentation format and how it's structured.

## Data Structure

### Batch Format

YOLO instance segmentation data is organized in batches with the following structure:

```python
batch = {
    'img': torch.Tensor,        # Input images [B, C, H, W]
    'masks': List[torch.Tensor], # Instance masks (one per batch item)
    'cls': torch.Tensor,        # Class labels for each instance [N]
    'batch_idx': torch.Tensor,  # Batch index for each instance [N]
    'bboxes': torch.Tensor,     # Bounding boxes [N, 4] (optional)
    # ... other metadata
}
```

### Key Components

#### 1. Instance Masks (`masks`)

- **Type**: `List[torch.Tensor]`
- **Structure**: List of 2D tensors, one per batch item
- **Shape**: Each mask is `[H, W]` where H, W are mask dimensions
- **Values**: 
  - `0` = Background pixels
  - `1, 2, 3, ...` = Instance IDs for different object instances

**Example:**
```python
# For batch_size=2:
masks = [
    torch.tensor([[0, 1, 1], [0, 2, 0]]),  # Batch item 0: 2 instances
    torch.tensor([[0, 0, 3], [3, 3, 0]])   # Batch item 1: 1 instance
]
```

#### 2. Class Labels (`cls`)

- **Type**: `torch.Tensor`
- **Shape**: `[N]` where N = total number of instances across all batch items
- **Values**: Integer class indices (e.g., 0-79 for COCO dataset)

**Example:**
```python
cls = torch.tensor([5, 12, 3])  # Instance 1→class 5, Instance 2→class 12, Instance 3→class 3
```

#### 3. Batch Index (`batch_idx`)

- **Type**: `torch.Tensor`
- **Shape**: `[N]` where N = total number of instances
- **Values**: Which batch item each instance belongs to

**Example:**
```python
batch_idx = torch.tensor([0, 0, 1])  # First 2 instances in batch item 0, last instance in batch item 1
```

## Complete Example

### Input Data
```python
batch = {
    'masks': [
        torch.tensor([[0, 1, 2], [0, 1, 0]]),  # Batch item 0
        torch.tensor([[0, 0, 3], [3, 3, 0]])   # Batch item 1
    ],
    'cls': torch.tensor([5, 12, 3]),           # Classes for instances 1, 2, 3
    'batch_idx': torch.tensor([0, 0, 1])      # Instance→batch mapping
}
```

### Interpretation
- **Batch item 0**: 
  - Instance 1 (pixels with value 1) → Class 5 (e.g., "bus")
  - Instance 2 (pixels with value 2) → Class 12 (e.g., "stop sign")
- **Batch item 1**:
  - Instance 3 (pixels with value 3) → Class 3 (e.g., "motorcycle")

## Key Characteristics

### 1. Instance-Centric Design
- Each object instance gets a unique ID within its batch item
- Multiple instances of the same class have different IDs
- Enables tracking and counting individual objects

### 2. Sparse Representation
- Only foreground pixels are explicitly labeled
- Background pixels are always 0
- Efficient storage for scenes with few objects

### 3. Variable Instance Count
- Different batch items can have different numbers of instances
- `cls` and `batch_idx` arrays grow/shrink based on total instances
- Flexible for real-world scenarios

## Differences from Semantic Segmentation

| Aspect | YOLO Instance Format | Semantic Segmentation |
|--------|---------------------|---------------------|
| **Pixel Values** | Instance IDs (1,2,3...) | Class labels (0-79) |
| **Multiple Objects** | Different IDs per instance | Same class label |
| **Background** | Always 0 | Often 255 (ignore) |
| **Data Structure** | List of masks + metadata | Single dense mask |
| **Use Case** | Object detection + segmentation | Scene understanding |

## Usage in Training Pipeline

### 1. Data Loading
```python
# Ultralytics dataloader provides:
for batch in dataloader:
    images = batch['img']           # [B, 3, H, W]
    instance_masks = batch['masks'] # List of [H, W] tensors
    classes = batch['cls']          # [N] class indices
    batch_indices = batch['batch_idx'] # [N] batch mapping
```

### 2. Loss Computation
```python
# For instance segmentation loss:
def compute_instance_loss(predictions, batch):
    # predictions: [B, num_classes, H, W]
    # batch: YOLO format
    
    # Extract instance masks and convert to appropriate format
    instance_masks = batch['masks']
    classes = batch['cls']
    
    # Compute loss per instance...
```

### 3. Conversion to Semantic Format
```python
# When needed for semantic segmentation:
from src.utils.mask_processing import convert_instance_masks_to_semantic

semantic_masks = convert_instance_masks_to_semantic(batch)
# Result: [B, H, W] with class labels per pixel
```

## File Format on Disk

### Label Files (.txt)
YOLO instance segmentation annotations are stored as text files:

```
# Format: class_id x1 y1 x2 y2 ... xn yn
0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4  # Polygon for instance of class 0
1 0.5 0.6 0.7 0.6 0.7 0.8 0.5 0.8  # Polygon for instance of class 1
```

- Each line represents one instance
- Coordinates are normalized (0-1)
- Polygons define the instance boundary

### Dataset Structure
```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

## Advanced Features

### 1. Multi-Scale Masks
- Masks can have different resolutions
- Automatically padded/resized during batch processing
- Supports efficient training with variable input sizes

### 2. Memory Optimization
- Sparse mask representation
- Only stores non-zero regions
- Compressed format for large datasets

### 3. Augmentation Support
- Geometric transformations (rotation, scaling)
- Color space modifications
- Mosaic and mixup augmentations

## Common Issues and Solutions

### 1. Misaligned Instance/Class Arrays
**Problem**: More instances in masks than class labels
```python
# Problematic:
masks = [torch.tensor([[0, 1, 2, 3]])]  # 3 instances
cls = torch.tensor([5, 12])             # Only 2 classes
```

**Solution**: Validate data consistency during preprocessing

### 2. Instance ID Gaps
**Problem**: Non-consecutive instance IDs
```python
# Avoid:
mask = torch.tensor([[0, 1, 5, 7]])  # Gaps: no instances 2,3,4,6
```

**Solution**: Use consecutive IDs starting from 1

### 3. Large Instance IDs
**Problem**: Instance IDs that are too large
```python
# Problematic:
mask = torch.tensor([[0, 1, 999]])  # ID 999 might cause indexing issues
```

**Solution**: Keep instance IDs small and consecutive

## Best Practices

1. **Validate Data Consistency**
   - Ensure `len(unique_instances) == len(cls[batch_idx == i])`
   - Check for gaps in instance IDs

2. **Handle Edge Cases**
   - Empty masks (no instances)
   - Single-pixel instances
   - Overlapping instances

3. **Memory Management**
   - Use appropriate data types (`uint8` for masks)
   - Implement efficient batching
   - Consider mask compression for large datasets

4. **Quality Control**
   - Visualize masks during development
   - Validate class label ranges
   - Check spatial consistency

## References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [Instance Segmentation Papers](https://paperswithcode.com/task/instance-segmentation)

---

*This document describes the YOLO instance segmentation format as used in the Ultralytics framework and compatible systems.*
