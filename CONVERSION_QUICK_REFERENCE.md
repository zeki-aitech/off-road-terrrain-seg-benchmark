# Quick Reference: Instance-to-Semantic Conversion

## Summary
Converts YOLO instance segmentation masks (instance IDs) to semantic segmentation masks (class IDs).

## Key Transformation
```python
# INPUT: Instance mask with class mapping
instance_mask = [[0, 1, 2], [0, 1, 0]]  # Instance IDs
cls = [3, 7]                             # Class labels for instances 1,2

# OUTPUT: Semantic mask
semantic_mask = [[255, 3, 7], [255, 3, 255]]  # Class IDs directly
```

## Critical Index Mapping
- **Instance ID 1** → `cls[0]` (first class label)
- **Instance ID 2** → `cls[1]` (second class label)
- **Instance ID 0** → `ignore_index` (background)

## Function Signature
```python
def convert_instance_masks_to_semantic(masks, cls, batch_idx, ignore_index=255, num_classes=None):
    """
    Args:
        masks: List of tensors with instance IDs (0=bg, 1,2,3...=instances)
        cls: Tensor of class labels for all instances
        batch_idx: Tensor mapping instances to batch items
        ignore_index: Value for background/invalid pixels (default: 255)
        num_classes: Optional validation for class range
    
    Returns:
        List of semantic masks with class IDs per pixel
    """
```

## Why This Conversion is Needed

### YOLO Format (Instance-Based)
- **Purpose:** Object detection + segmentation
- **Pixel values:** Instance IDs (1, 2, 3, ...)
- **Class info:** Separate `cls` tensor
- **Instance-aware:** Can distinguish between multiple objects of same class

### Semantic Format (Class-Based)
- **Purpose:** Pixel-wise classification
- **Pixel values:** Class IDs (0, 1, 2, ..., num_classes-1)
- **Class info:** Embedded in pixel values
- **Instance-agnostic:** All pixels of same class have same value

## Training Requirements
Most semantic segmentation models and metrics expect:
- Direct class labels per pixel
- Background as `ignore_index` (usually 255)
- Cross-entropy loss: `F.cross_entropy(pred, target, ignore_index=255)`

## Validation Notes
- Empty batches → return tensor with `ignore_index`
- Invalid class values → map to `ignore_index`
- Out-of-bounds instances → map to `ignore_index`
- Dimension consistency maintained across batch
