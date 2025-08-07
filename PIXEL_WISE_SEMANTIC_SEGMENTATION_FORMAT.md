# Pixel-wise Semantic Segmentation Format

This document explains the pixel-wise semantic segmentation format used in computer vision tasks.

## Overview

Pixel-wise semantic segmentation format assigns each pixel in an image to a specific semantic class. Every pixel contains a class index that directly corresponds to the object category at that spatial location.

## Data Structure

### Tensor Shape
- **Single Image**: `[H, W]` - Height × Width
- **Batch**: `[B, H, W]` - Batch × Height × Width

### Data Type
- **Typical**: `torch.long` or `torch.uint8`
- **Range**: Integer values representing class indices

## Pixel Value Encoding

### Class Index Mapping
Each pixel value directly represents a semantic class:

```
0 = Class 0 (often background or first class)
1 = Class 1 (e.g., person)
2 = Class 2 (e.g., car)
3 = Class 3 (e.g., bicycle)
...
N = Class N (last class in dataset)
255 = ignore_index (pixels to ignore during training/evaluation)
```

### Special Values
- **ignore_index (255)**: Pixels that should be ignored during loss calculation and evaluation
- **Valid range**: Typically 0 to num_classes-1, plus 255 for ignore

## Format Properties

### 1. Direct Class Assignment
- Each pixel contains exactly one class index
- No ambiguity - one pixel, one class
- Spatially dense classification

### 2. Semantic Focus
- Represents **what** is at each location
- No distinction between different instances of the same class
- Pure categorical classification per pixel

### 3. Loss Compatibility
- Direct input to `nn.CrossEntropyLoss`
- Compatible with standard segmentation metrics (mIoU, pixel accuracy)
- No preprocessing required for training

## Concrete Example

```python
# Example semantic mask for a 4x4 image
semantic_mask = torch.tensor([
    [255, 255,   1,   1],  # ignore, ignore, person, person
    [255,   2,   2,   1],  # ignore, car, car, person
    [  0,   2,   2,   0],  # background, car, car, background
    [  0,   0,   3,   3]   # background, background, bike, bike
])
```

In this example:
- Class 0: Background
- Class 1: Person
- Class 2: Car  
- Class 3: Bicycle
- Value 255: Ignore regions

---

*This format serves as the standard representation for semantic segmentation tasks across most computer vision frameworks and datasets.*
