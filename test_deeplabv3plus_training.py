#!/usr/bin/env python3
"""
Full DeepLabV3+ Training Test Script

This script demonstrates end-to-end training of the DeepLabV3+ model with the Ultralytics framework.
It runs a short training session to verify that the complete pipeline works correctly.
"""

import sys
import os
import torch
from pathlib import Path

# Set project root and apply monkey patches
project_root = '/workspaces/off-road-terrrain-seg-benchmark'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# CRITICAL: Apply monkey patches BEFORE any Ultralytics imports
from src.patches import apply_patches
apply_patches()

# Now safe to import everything
from ultralytics.utils import DEFAULT_CFG_DICT
from src.models.deeplabv3plus.train import DeepLabV3PlusSemanticSegmentationTrainer


def run_deeplabv3plus_training():
    """Run a test training session with DeepLabV3+ model."""
    
    print("="*60)
    print("🚀 DEEPLABV3+ TRAINING TEST")
    print("="*60)
    
    # Auto-detect best available device
    if torch.cuda.is_available():
        device = 'cuda'
        batch_size = 8  # Larger batch for GPU
        image_size = 320  # Larger image size for GPU
        print("🚀 GPU detected - using CUDA acceleration")
    else:
        device = 'cpu'
        batch_size = 4  # Smaller batch for CPU
        image_size = 160  # Smaller image size for CPU
        print("💻 Using CPU (GPU not available)")
    
    # Training configuration
    train_config = {
        'model': '/workspaces/off-road-terrrain-seg-benchmark/src/cfg/models/deeplabv3plus_resnet50.yaml',
        'data': 'coco8-seg.yaml',  # Use standard COCO8 segmentation dataset
        'epochs': 3,  # Short test run
        'batch': batch_size,   # Auto-adjust batch size based on device
        'imgsz': image_size, # Auto-adjust image size based on device
        'device': device,  # Auto-select device
        'workers': 4 if device == 'cuda' else 1,  # More workers for GPU
        'project': f'{project_root}/runs/test_training',
        'name': f'deeplabv3plus_coco8_{device}_test',
        'exist_ok': True,
        'verbose': True,
        'save': True,
        'plots': False,  # Disable plots for cleaner output
        'val': True,     # Enable validation
        'lr0': 0.001,    # Learning rate
        'optimizer': 'Adam',
    }
    
    print("📋 Training Configuration:")
    for key, value in train_config.items():
        print(f"   {key}: {value}")
    
    # Detailed device information
    print(f"\n🔧 Device Information:")
    print(f"   Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"   CPU: {torch.get_num_threads()} threads")
    print(f"   PyTorch: {torch.__version__}")
    
    try:
        print("\n📦 Creating trainer...")
        trainer = DeepLabV3PlusSemanticSegmentationTrainer(overrides=train_config)
        
        print("🏗️  Setting up model...")
        model = trainer.get_model(cfg=train_config['model'])
        print(f"✅ Model created: {type(model).__name__}")
        
        print("\n🚀 Starting training...")
        print("   Note: This is a test run with minimal epochs")
        print("   For production, use 50-100 epochs with larger batch sizes")
        print("   Metrics: mIoU, PixelAcc, MeanAcc (semantic segmentation metrics)")
        print("   Not using mAP (object detection metric) - inappropriate for semantic segmentation")
        
        # Start training
        results = trainer.train()
        
        print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Results saved to: {trainer.save_dir}")
        
        if results:
            print("📈 Training Results:")
            # Semantic segmentation metrics (not mAP which is for object detection)
            if hasattr(results, 'miou'):
                print(f"   mIoU: {results.miou:.4f}")
            if hasattr(results, 'pixel_acc'):
                print(f"   Pixel Accuracy: {results.pixel_acc:.4f}")
            if hasattr(results, 'mean_acc'):
                print(f"   Mean Accuracy: {results.mean_acc:.4f}")
            if hasattr(results, 'fitness'):
                print(f"   Fitness: {results.fitness:.4f}")
            
            # If no specific semantic metrics, show what's available
            if not any(hasattr(results, attr) for attr in ['miou', 'pixel_acc', 'mean_acc']):
                print("   Note: Semantic segmentation metrics not directly accessible in results object")
                print("   Check validation logs above for mIoU, PixelAcc, and MeanAcc values")
        
        print("\n🎉 DeepLabV3+ training pipeline is fully functional!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        print("\n📋 Full error traceback:")
        traceback.print_exc()
        return False


def main():
    """Main function to run the training test."""
    
    # Check if we can access COCO8 dataset (it should be automatically downloaded)
    print("📊 Using COCO8 segmentation dataset for testing...")
    print("   This dataset will be automatically downloaded if not present")
    
    # Run training test
    success = run_deeplabv3plus_training()
    
    if success:
        print("\n" + "="*60)
        print("🎊 SUCCESS! DeepLabV3+ training pipeline works perfectly!")
        print("="*60)
        print("\n📋 NEXT STEPS:")
        print("1. 🎯 Increase epochs (50-100) for real training")
        print("2. 📊 Use larger batch sizes (4-16) for better performance")
        print("3. 🖼️  Use full resolution images (640x640)")
        print("4. 🔧 Fine-tune hyperparameters for your dataset")
        print("5. 🚀 Deploy to GPU for faster training")
        
    else:
        print("\n" + "="*60)
        print("⚠️  Training test failed - check errors above")
        print("="*60)
    
    return success


if __name__ == "__main__":
    main()
