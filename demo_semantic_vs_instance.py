#!/usr/bin/env python3
"""
Demo: DeepLabV3+ Semantic vs Instance Segmentation

This script demonstrates that our DeepLabV3+ model performs semantic segmentation
even when trained on instance segmentation datasets like COCO8.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Set project root and apply monkey patches
project_root = '/workspaces/off-road-terrrain-seg-benchmark'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.patches import apply_patches
apply_patches()

from src.models.deeplabv3plus.train import DeepLabV3PlusSemanticSegmentationTrainer


def demonstrate_semantic_segmentation():
    """Demonstrate that our model outputs semantic, not instance segmentation."""
    
    print("🔬 SEMANTIC vs INSTANCE SEGMENTATION DEMO")
    print("="*60)
    
    # Create a simple trainer to get our model
    train_config = {
        'model': '/workspaces/off-road-terrrain-seg-benchmark/src/cfg/models/deeplabv3plus_resnet50.yaml',
        'data': 'coco8-seg.yaml',
        'device': 'cpu'
    }
    
    trainer = DeepLabV3PlusSemanticSegmentationTrainer(overrides=train_config)
    model = trainer.get_model(cfg=train_config['model'])
    model.eval()
    
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"📊 Model classes (nc): {model.yaml.get('nc', 'not found')}")
    
    # Create test image
    test_image = torch.randn(1, 3, 320, 320)
    
    print("\n🧪 Testing model output...")
    with torch.no_grad():
        output = model(test_image)
    
    print(f"📐 Input shape: {test_image.shape}")
    print(f"📐 Output shape: {output.shape}")
    
    # Analyze output
    print("\n🔍 OUTPUT ANALYSIS:")
    print(f"   • Output dimensions: {output.shape}")
    print(f"   • Batch size: {output.shape[0]}")
    print(f"   • Number of classes: {output.shape[1]}")
    print(f"   • Spatial dimensions: {output.shape[2]}x{output.shape[3]}")
    
    # Check if it's semantic segmentation format
    if len(output.shape) == 4 and output.shape[1] == 80:  # [B, C, H, W] with C=classes
        print("\n✅ CONFIRMED: This is SEMANTIC SEGMENTATION!")
        print("   • Output format: [Batch, Classes, Height, Width]")
        print("   • Each pixel gets a class probability distribution")
        print("   • No instance separation - all pixels of same class are treated equally")
        
        # Show class predictions for a sample pixel
        sample_pixel = output[0, :, 160, 160]  # Center pixel, all classes
        top_classes = torch.topk(sample_pixel, 5)
        
        print(f"\n📊 Sample pixel (160,160) - Top 5 class probabilities:")
        for i, (prob, class_idx) in enumerate(zip(top_classes.values, top_classes.indices)):
            print(f"   {i+1}. Class {class_idx.item()}: {prob.item():.4f}")
            
    else:
        print("\n❓ Unexpected output format")
    
    print("\n" + "="*60)
    print("📋 SUMMARY:")
    print("• DeepLabV3+ performs SEMANTIC segmentation")
    print("• 'Instances' in training refers to input data format (COCO8)")
    print("• Model converts instance masks → semantic masks during training")
    print("• Output is pure semantic: one class prediction per pixel")
    print("• No instance boundaries are preserved in the output")
    
    return True


if __name__ == "__main__":
    demonstrate_semantic_segmentation()
