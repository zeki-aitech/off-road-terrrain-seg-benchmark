# Complete Guide: Implementing Custom Models Compatible with Ultralytics

This guide provides a comprehensive, step-by-step approach to integrate any custom deep learning model with the Ultralytics framework. Whether you're implementing advanced architectures like DeepLabV3+, U-Net, Vision Transformers, custom detection heads, or entirely novel architectures, this guide will help you achieve full compatibility with Ultralytics' training, validation, and prediction APIs.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Step 1: Model Architecture](#step-1-model-architecture)
4. [Step 2: Custom Components](#step-2-custom-components)
5. [Step 3: Monkey Patching](#step-3-monkey-patching)
6. [Step 4: Unified Interface](#step-4-unified-interface)
7. [Step 5: Testing & Validation](#step-5-testing--validation)
8. [Best Practices](#best-practices)
9. [Common Issues & Solutions](#common-issues--solutions)

## Overview

Integrating a custom model with Ultralytics requires implementing several key components:
- **Model Architecture**: Define your custom neural network modules and layers
- **Model Configuration**: Create YAML config files defining your architecture  
- **Trainer**: Handle training logic, loss computation, and optimization
- **Validator**: Implement validation logic and custom metrics
- **Predictor**: Handle inference, pre/post-processing, and result formatting
- **Monkey Patching**: Integrate with Ultralytics' model parsing system
- **Unified Interface**: Create a YOLO-like API for seamless usage

This guide supports all Ultralytics tasks: **segmentation**, **detection**, and **classification**.

## Project Structure

```
your_project/
├── src/
│   ├── __init__.py
│   ├── patches.py                 # Monkey patching for Ultralytics
│   ├── cfg/
│   │   └── models/
│   │       └── your_model.yaml    # Model configuration
│   ├── models/
│   │   └── your_model/
│   │       ├── __init__.py
│   │       ├── model.py           # Unified interface
│   │       ├── train.py           # Custom trainer
│   │       ├── val.py             # Custom validator
│   │       └── predict.py         # Custom predictor
│   ├── nn/
│   │   ├── __init__.py
│   │   ├── tasks.py               # Model task definition
│   │   └── modules/
│   │       ├── __init__.py
│   │       ├── block.py           # Custom blocks/layers
│   │       └── head.py            # Custom heads
│   └── utils/
│       ├── __init__.py
│       ├── loss.py                # Custom loss functions
│       └── metrics.py             # Custom metrics
└── tests/
    ├── test_compatibility.py      # Compatibility tests
    ├── test_training.py           # Training tests
    └── test_predictor.py          # Predictor tests
```

## Step 1: Model Architecture

### 1.1 Define Model Configuration (YAML)

Create `src/cfg/models/your_model.yaml`:

```yaml
# Custom Model Configuration
backbone:
  # [from, repeats, module, args]
  - [-1, 1, YourCustomStem, [3, 64]]           # Custom stem layer
  - [-1, 1, YourCustomBlock, [64, 128, 2]]     # Custom blocks
  - [-1, 3, YourCustomBlock, [128, 256, 2]]    # Repeated blocks
  - [-1, 1, YourCustomASPP, [256, 512]]        # Custom feature extraction

# Custom Model Head
head:
  - [-1, 1, Upsample, [None, 2, 'nearest']]
  - [[-1, 2], 1, Concat, [1]]                  # Feature concatenation
  - [-1, 3, YourCustomConv, [512, 256]]        # Custom convolutions
  - [-1, 1, YourCustomHead, [256, nc]]         # Custom prediction head

# Model metadata
nc: 80  # number of classes (adjust for your task)
scales: # model compound scaling constants (optional)
  n: [0.33, 0.25, 2.0]  # depth, width, max_channels
  s: [0.33, 0.50, 2.0]
  m: [0.67, 0.75, 3.0]
  l: [1.00, 1.00, 4.0]
  x: [1.00, 1.25, 6.0]
```

### 1.2 Implement Custom Modules

Create `src/nn/modules/block.py`:

```python
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

class YourCustomStem(nn.Module):
    """Custom stem module for initial feature extraction."""
    def __init__(self, c1, c2):
        super().__init__()
        # Example implementations for different architectures:
        
        # For CNN-based models: Large kernel conv + pooling
        self.stem = nn.Sequential(
            Conv(c1, c2, 7, 2, 3),  # Large kernel for initial feature extraction
            nn.MaxPool2d(3, 2, 1)   # Spatial downsampling
        )
        
        # For Vision Transformer: Patch embedding
        # self.stem = nn.Conv2d(c1, c2, 16, 16)  # 16x16 patch embedding
        
        # For lightweight models: Depthwise separable conv
        # self.stem = nn.Sequential(
        #     nn.Conv2d(c1, c1, 3, 2, 1, groups=c1),  # Depthwise
        #     nn.Conv2d(c1, c2, 1, 1, 0)              # Pointwise
        # )
    
    def forward(self, x):
        return self.stem(x)

class YourCustomBlock(nn.Module):
    """Custom building block for your architecture."""
    def __init__(self, c1, c2, stride=1):
        super().__init__()
        # Example implementations for different architectures:
        
        # For ResNet-style: Residual block
        self.conv1 = Conv(c1, c2, 3, stride, 1)
        self.conv2 = Conv(c2, c2, 3, 1, 1)
        self.shortcut = Conv(c1, c2, 1, stride) if stride != 1 or c1 != c2 else nn.Identity()
        
        # For DenseNet-style: Dense block  
        # self.layers = nn.ModuleList([Conv(c1 + i*growth_rate, growth_rate, 3, 1, 1) 
        #                             for i in range(num_layers)])
        
        # For Transformer-style: Multi-head attention block
        # self.attention = nn.MultiheadAttention(c1, num_heads=8)
        # self.ffn = nn.Sequential(Conv(c1, c2, 1), nn.ReLU(), Conv(c2, c1, 1))
        
    def forward(self, x):
        # Example: Residual connection
        identity = self.shortcut(x)
        out = self.conv2(self.conv1(x))
        return out + identity
```

### 1.3 Implement Custom Head

Create `src/nn/modules/head.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class YourCustomHead(nn.Module):
    """Custom prediction head for your specific task."""
    def __init__(self, c1, nc):
        super().__init__()
        self.nc = nc  # number of classes
        self.task = 'segment'  # or 'detect', 'classify'
        
        # Task-specific head implementations:
        
        # For segmentation: Simple conv to class logits
        if self.task == 'segment':
            self.conv = nn.Conv2d(c1, nc, 1)
        
        # For detection: YOLO-style head with bbox + confidence
        elif self.task == 'detect':
            self.conv = nn.Conv2d(c1, nc + 5, 1)  # classes + box + conf
        
        # For classification: Global pooling + linear
        elif self.task == 'classify':
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(c1, nc)
            
    def forward(self, x):
        if self.task == 'segment':
            # Segmentation: return class logits per pixel
            return self.conv(x)
            
        elif self.task == 'detect':
            # Detection: return detections [batch, anchors, classes+5]
            return self.conv(x).permute(0, 2, 3, 1)
            
        elif self.task == 'classify':
            # Classification: return class probabilities
            x = self.pool(x).flatten(1)
            return self.fc(x)
        
    def forward(self, x):
        return self.conv(x)  # This will be task-specific based on head implementation
```

### 1.4 Define Model Task

Create `src/nn/tasks.py`:

```python
from ultralytics.nn.tasks import SegmentationModel, DetectionModel, ClassificationModel
from ultralytics.utils import yaml_load
from ultralytics.nn.modules import Conv, Upsample, Concat
from .modules.block import YourCustomStem, YourCustomBlock
from .modules.head import YourCustomHead

class YourCustomModel(SegmentationModel):  # Choose: SegmentationModel, DetectionModel, ClassificationModel
    """Your custom model extending Ultralytics base."""
    
    def __init__(self, cfg='your_model.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
    
    @staticmethod
    def _from_yaml(cfg):
        """Create model from YAML configuration."""
        # Parse YAML and create model
        return YourCustomModel(cfg)
    
    def forward(self, x, augment=False, profile=False, visualize=False):
        """Forward pass through the model."""
        # You can override this for custom forward logic
        # or rely on the parent class implementation
        return super().forward(x, augment, profile, visualize)
```

## Step 2: Custom Components

### 2.1 Custom Trainer

Create `src/models/your_model/train.py`:

```python
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.models.yolo.detect import DetectionTrainer  
from ultralytics.models.yolo.classify import ClassificationTrainer
from ultralytics.utils import DEFAULT_CFG
from .val import YourCustomValidator

class YourCustomTrainer(SegmentationTrainer):  # Choose: SegmentationTrainer, DetectionTrainer, ClassificationTrainer
    """Custom trainer for your model."""
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)
    
    def get_validator(self):
        """Return custom validator."""
        self.loss_names = ['loss']  # Define loss names for your task
        return YourCustomValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args)
        )
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get your custom model."""
        from src.nn.tasks import YourCustomModel
        model = YourCustomModel(cfg, nc=self.data["nc"], verbose=verbose)
        if weights:
            model.load(weights)
        return model
    
    def criterion(self, preds, batch):
        """Compute loss for your specific task."""
        # Implement your custom loss computation
        from src.utils.loss import YourCustomLoss
        if not hasattr(self, 'compute_loss'):
            self.compute_loss = YourCustomLoss()
        return self.compute_loss(preds, batch)
    
    def label_loss_items(self, loss_items=None, prefix="train"):
        """Return labeled loss items for logging."""
        if loss_items is None:
            return ['loss']  # Return default loss names
        return [f"{prefix}/{name}" for name in self.loss_names]
```

### 2.2 Custom Validator

Create `src/models/your_model/val.py`:

```python
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.classify import ClassificationValidator
from ultralytics.utils import DEFAULT_CFG
from src.utils.metrics import YourCustomMetrics

class YourCustomValidator(SegmentationValidator):  # Choose: SegmentationValidator, DetectionValidator, ClassificationValidator
    """Custom validator for your model."""
    
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.metrics = YourCustomMetrics()
    
    def init_metrics(self, model):
        """Initialize metrics for your specific task."""
        self.nc = model.nc
        self.metrics = YourCustomMetrics(
            save_dir=self.save_dir,
            plot=self.args.plots,
            names=model.names
        )
    
    def get_desc(self):
        """Return description string for progress bar."""
        # Customize based on your task:
        # Segmentation: mIoU, PixelAcc, MeanAcc
        # Detection: mAP50, mAP50-95, Precision, Recall
        # Classification: Accuracy, Top-5 Accuracy
        return ('%22s' + '%11s' * 4) % ('classes', 'metric1', 'metric2', 'metric3')
    
    def postprocess(self, preds):
        """Postprocess predictions for your task."""
        # Implement task-specific postprocessing
        return preds
    
    def update_metrics(self, preds, batch):
        """Update metrics with predictions and ground truth."""
        # Update your custom metrics
        self.metrics.update(preds, batch)
    
    def finalize_metrics(self, *args, **kwargs):
        """Finalize and return metrics."""
        return self.metrics.finalize()
```

### 2.3 Custom Predictor

Create `src/models/your_model/predict.py`:

```python
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.models.yolo.classify import ClassificationPredictor
from ultralytics.utils import DEFAULT_CFG
from ultralytics.engine.results import Results
import torch
import numpy as np

class YourCustomPredictor(SegmentationPredictor):  # Choose: SegmentationPredictor, DetectionPredictor, ClassificationPredictor
    """Custom predictor for your model."""
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
    
    def postprocess(self, preds, img, orig_imgs):
        """Postprocess predictions for your specific task."""
        # Handle different input formats
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        
        if not isinstance(preds, torch.Tensor):
            preds = torch.tensor(preds)
        
        # Ensure correct dimensions [B, C, H, W]
        if preds.dim() == 3:
            preds = preds.unsqueeze(0)
        
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            
            # Apply your custom postprocessing logic
            processed_pred = self.custom_postprocess(pred, orig_img.shape[:2])
            
            # Create Results object appropriate for your task
            result = Results(
                orig_img=orig_img,
                path=getattr(self, 'source', ''),
                names=getattr(self.model, 'names', {}),
                # Task-specific outputs:
                masks=processed_pred if self.task == 'segment' else None,
                boxes=processed_pred if self.task == 'detect' else None,
                probs=processed_pred if self.task == 'classify' else None
            )
            results.append(result)
        
        return results
    
    def custom_postprocess(self, pred, orig_shape):
        """Your custom postprocessing logic for your specific task."""
        # Example for segmentation:
        if self.task == 'segment':
            processed = torch.softmax(pred, dim=0)
            processed = torch.argmax(processed, dim=0)
        
        # Example for detection:
        elif self.task == 'detect':
            # Apply NMS, confidence thresholding, etc.
            from ultralytics.utils.ops import non_max_suppression
            processed = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
        
        # Example for classification:
        elif self.task == 'classify':
            processed = torch.softmax(pred, dim=-1)
        
        # Resize to original image size if needed
        if hasattr(processed, 'shape') and len(processed.shape) == 2:
            if processed.shape != orig_shape:
                processed = torch.nn.functional.interpolate(
                    processed.unsqueeze(0).unsqueeze(0).float(),
                    size=orig_shape,
                    mode='nearest'
                ).squeeze().long()
        
        return processed.cpu().numpy().astype(np.uint8)
```

## Step 3: Monkey Patching

### 3.1 Create Patches Module

Create `src/patches.py`:

```python
"""
Monkey patches for Ultralytics to support custom models.
This module must be imported before any Ultralytics imports.
"""

def patch_parse_model():
    """Patch Ultralytics parse_model to recognize custom modules."""
    from ultralytics.nn.tasks import parse_model as original_parse_model
    from ultralytics.nn import tasks
    import sys
    
    def custom_parse_model(d, ch, verbose=True, imgsz=640):
        """Custom parse_model that includes your modules."""
        
        # Import custom modules within the function to avoid circular imports
        try:
            from src.nn.modules.block import YourCustomStem, YourCustomBlock
            from src.nn.modules.head import YourCustomHead
            
            # Add custom modules to the current module's globals for eval() to find them
            current_globals = sys.modules['ultralytics.nn.tasks'].__dict__
            
            # Register custom modules in ultralytics.nn.tasks namespace
            custom_modules = {
                'YourCustomStem': YourCustomStem,
                'YourCustomBlock': YourCustomBlock,
                'YourCustomHead': YourCustomHead,
            }
            
            for name, module_class in custom_modules.items():
                if name not in current_globals:
                    current_globals[name] = module_class
                    
        except ImportError as e:
            if verbose:
                print(f"Warning: Could not import custom modules: {e}")
        
        # Call original parse_model with updated globals
        return original_parse_model(d, ch, verbose, imgsz)
    
    # Replace the original function
    tasks.parse_model = custom_parse_model
    
    print("✅ Monkey patches applied: parse_model replaced with custom version")

# Apply patches when module is imported
patch_parse_model()
```

### 3.2 Real Working Example from Successful Implementation

Here's a proven working pattern adapted from a successful custom model integration:

```python
"""
Monkey patches for Ultralytics to support custom models.
This module must be imported before any Ultralytics imports.
"""

def patch_parse_model():
    """Patch Ultralytics parse_model to recognize custom modules."""
    from ultralytics.nn.tasks import parse_model as original_parse_model
    from ultralytics.nn import tasks
    import sys
    
    def enhanced_parse_model(d, ch, verbose=True, imgsz=640):
        """Enhanced parse_model that includes custom modules."""
        
        # Import custom modules inside function to avoid circular imports
        try:
            # Import all your custom modules here
            from src.nn.modules.block import (
                YourCustomStem, YourCustomBlock, YourCustomASPP  # Example modules
            )
            from src.nn.modules.head import YourCustomHead
            
            # Additional modules for different architectures:
            # from src.nn.modules.attention import MultiHeadAttention
            # from src.nn.modules.transformer import TransformerBlock
            # from src.nn.modules.convnext import ConvNeXtBlock
            
            # Get the ultralytics.nn.tasks module's global namespace
            # This is where eval() looks for module names during parsing
            tasks_globals = sys.modules['ultralytics.nn.tasks'].__dict__
            
            # Register all your custom modules in that namespace
            custom_modules = {
                'YourCustomStem': YourCustomStem,
                'YourCustomBlock': YourCustomBlock, 
                'YourCustomASPP': YourCustomASPP,
                'YourCustomHead': YourCustomHead,
                # Add all your custom modules here:
                # 'MultiHeadAttention': MultiHeadAttention,
                # 'TransformerBlock': TransformerBlock,
                # 'ConvNeXtBlock': ConvNeXtBlock,
            }
            
            for name, module_class in custom_modules.items():
                if name not in tasks_globals:
                    tasks_globals[name] = module_class
                    
        except ImportError as e:
            if verbose:
                print(f"Warning: Could not import custom modules: {e}")
        
        # Call the original parse_model function
        return original_parse_model(d, ch, verbose, imgsz)
    
    # Replace the parse_model function
    tasks.parse_model = enhanced_parse_model
    
    print("✅ Monkey patches applied: parse_model enhanced with custom modules")

# Apply patches immediately when module is imported
patch_parse_model()
```

### 3.3 Key Points for Correct Monkey Patching

1. **Import modules inside the function**: Avoids circular import issues
2. **Target the right namespace**: `sys.modules['ultralytics.nn.tasks'].__dict__` is where `eval()` looks
3. **Handle import errors gracefully**: Use try/catch for robustness
4. **Apply patches on import**: Call the patch function immediately

### 3.4 Import Patches Early

In your main scripts, always import patches first:

```python
import sys
import os

# Add src to path FIRST
sys.path.insert(0, '/path/to/your/src')

# Import patches BEFORE any Ultralytics imports
import patches  # This applies the monkey patches

# Now safe to import Ultralytics and your models
from ultralytics import YOLO
from src.models.your_model.model import YourCustomModel
```

## Step 4: Unified Interface

### 4.1 Create Unified Model Class

Create `src/models/your_model/model.py`:

```python
from typing import Dict, Any
from ultralytics.engine.model import Model
from src.nn.tasks import YourCustomModel
from .train import YourCustomTrainer
from .val import YourCustomValidator
from .predict import YourCustomPredictor

class YourModel(Model):
    """Unified interface for your custom model."""
    
    def __init__(self, model="your_model.yaml", task="segment", verbose=False):
        """
        Initialize your custom model.
        
        Args:
            model (str): Path to model config or model name
            task (str): Task type (e.g., 'segment', 'detect', 'classify')
            verbose (bool): Verbose output
        """
        super().__init__(model=model, task=task, verbose=verbose)
    
    @property
    def task_map(self) -> Dict[str, Dict[str, Any]]:
        """Map tasks to their respective classes."""
        return {
            "segment": {
                "model": YourCustomModel,
                "trainer": YourCustomTrainer,
                "validator": YourCustomValidator,
                "predictor": YourCustomPredictor,
            },
            # Add other tasks if needed
            "detect": {
                "model": YourCustomModel,
                "trainer": YourCustomTrainer,
                "validator": YourCustomValidator,  
                "predictor": YourCustomPredictor,
            },
        }
```

### 4.2 Usage Example

```python
# Import patches first
import sys
sys.path.insert(0, '/path/to/your/src')
import patches

# Now use like YOLO
from src.models.your_model.model import YourModel

# Create model
model = YourModel("path/to/config.yaml", task="segment")

# Use exactly like YOLO
results = model.predict("image.jpg")
model.train(data="dataset.yaml", epochs=100)
metrics = model.val()
model.export(format="onnx")
```

## Step 5: Testing & Validation

### 5.1 Compatibility Test

Create `tests/test_compatibility.py`:

```python
import sys
sys.path.insert(0, '/path/to/your/src')
import patches

from src.models.your_model.model import YourModel
from ultralytics import YOLO

def test_model_compatibility():
    """Test model compatibility with Ultralytics."""
    
    # Test instantiation
    model = YourModel("config.yaml", task="segment")
    assert model is not None
    
    # Test methods exist
    assert hasattr(model, 'predict')
    assert hasattr(model, 'train')
    assert hasattr(model, 'val')
    assert hasattr(model, 'export')
    
    # Test task map
    task_map = model.task_map
    assert "segment" in task_map
    
    # Test prediction
    import numpy as np
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    results = model.predict(dummy_img, verbose=False)
    assert len(results) > 0

def test_vs_yolo():
    """Compare interface with standard YOLO."""
    model = YourModel("config.yaml")
    yolo = YOLO("yolo11n-seg.pt")
    
    # Compare methods
    model_methods = [m for m in dir(model) if not m.startswith('_')]
    yolo_methods = [m for m in dir(yolo) if not m.startswith('_')]
    
    essential_methods = ['predict', 'train', 'val', 'export']
    for method in essential_methods:
        assert method in model_methods
        assert method in yolo_methods

if __name__ == "__main__":
    test_model_compatibility()
    test_vs_yolo()
    print("✅ All compatibility tests passed!")
```

### 5.2 Training Test

Create `tests/test_training.py`:

```python
import sys
sys.path.insert(0, '/path/to/your/src')
import patches

from src.models.your_model.model import YourModel

def test_training():
    """Test training functionality."""
    
    model = YourModel("config.yaml", task="segment")
    
    # Test training with minimal parameters
    results = model.train(
        data='coco8-seg.yaml',
        epochs=2,
        batch=2,
        imgsz=320,
        device='cpu',
        verbose=False
    )
    
    assert results is not None
    print("✅ Training test passed!")

def test_validation():
    """Test validation functionality."""
    
    model = YourModel("config.yaml", task="segment")
    
    # Test validation
    results = model.val(
        data='coco8-seg.yaml',
        batch=2,
        imgsz=320,
        device='cpu',
        verbose=False
    )
    
    assert results is not None
    print("✅ Validation test passed!")

if __name__ == "__main__":
    test_training()
    test_validation()
    print("✅ All training tests passed!")
```

## Best Practices

### 1. **Module Organization**
- Keep custom modules in separate files
- Use clear, descriptive naming conventions
- Maintain consistent interfaces with Ultralytics

### 2. **Configuration Management**
- Use YAML files for model configuration
- Support different model scales (n, s, m, l, x)
- Include proper metadata (nc, scales, etc.)

### 3. **Error Handling**
- Add robust error handling in all components
- Provide clear error messages
- Handle edge cases (empty inputs, wrong dimensions)

### 4. **Testing Strategy**
- Test compatibility with Ultralytics
- Test training on standard datasets
- Test prediction with various input formats
- Compare performance with baseline models

### 5. **Documentation**
- Document all custom components
- Provide usage examples
- Include configuration guidelines
- Document any limitations or requirements

## Common Issues & Solutions

### 1. **Import Order Issues**
**Problem**: Custom modules not recognized during model parsing.
**Solution**: Always import patches before any Ultralytics imports.

```python
# ❌ Wrong order
from ultralytics import YOLO
import patches

# ✅ Correct order  
import patches
from ultralytics import YOLO
```

### 2. **Device Compatibility**
**Problem**: Model fails on different devices (CPU/GPU).
**Solution**: Handle device detection and tensor placement properly.

```python
def forward(self, x):
    device = x.device
    # Ensure all operations use the same device
    return self.layers(x.to(device))
```

### 3. **Metric Computation**
**Problem**: Metrics don't match expected task type.
**Solution**: Implement task-specific metrics and ensure proper metric keys.

```python
def finalize_metrics(self):
    def finalize_metrics(self):
        """Return metrics with correct keys for your task."""
        # Task-specific metrics:
        if self.task == 'segment':
            return {
                'metrics/mIoU': self.miou,
                'metrics/PixelAcc': self.pixel_acc,
                'fitness': self.fitness_score
            }
        elif self.task == 'detect':
            return {
                'metrics/mAP50': self.map50,
                'metrics/mAP50-95': self.map,
                'fitness': self.fitness_score
            }
        elif self.task == 'classify':
            return {
                'metrics/accuracy_top1': self.top1,
                'metrics/accuracy_top5': self.top5,
                'fitness': self.fitness_score
            }
```

### 4. **Model Export Issues**
**Problem**: Model fails to export to different formats.
**Solution**: Ensure all custom modules support torchscript/onnx export.

```python
class YourCustomModule(nn.Module):
    def forward(self, x):
        # Avoid dynamic operations that break export
        # Use static shapes where possible
        return self.process(x)
```

### 5. **Loss Computation**
**Problem**: Loss computation fails or produces NaN values.
**Solution**: Implement robust loss computation with proper tensor handling.

```python
def compute_loss(self, preds, targets):
    # Handle different prediction formats
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    
    # Ensure same device
    device = preds.device
    targets = targets.to(device)
    
    # Compute loss with proper shape handling
    loss = self.loss_fn(preds, targets)
    return loss
```

## Conclusion

This guide provides a complete framework for integrating custom models with Ultralytics. The key steps are:

1. **Design your model architecture** with proper YAML configuration
2. **Implement custom components** (trainer, validator, predictor)
3. **Apply monkey patches** to integrate with Ultralytics parsing
4. **Create a unified interface** that mimics YOLO's API
5. **Test thoroughly** to ensure compatibility and functionality

Following this guide ensures your custom model will work seamlessly with Ultralytics' ecosystem while maintaining all the framework's benefits like automatic logging, export functionality, and standardized APIs.

**✨ Benefits of This Approach:**
- **Full Ultralytics Compatibility**: Your model works exactly like YOLO
- **Automatic Logging**: MLflow, TensorBoard, Weights & Biases integration  
- **Export Support**: ONNX, TensorRT, CoreML, and more
- **Standardized APIs**: Consistent train/val/predict interface
- **Ecosystem Access**: Leverage all Ultralytics tools and utilities

---

**🎯 Key Takeaway**: The most critical aspect is ensuring proper import order (patches first) and implementing all required components with compatible interfaces. Once done correctly, your custom model will work exactly like any standard YOLO model, with access to the entire Ultralytics ecosystem!
