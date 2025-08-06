#!/usr/bin/env python3
"""
GPU-Optimized DeepLabV3+ Training Script

This script is optimized for GPU training with larger batch sizes and image resolutions.
It automatically detects GPU availability and uses optimal settings for GPU acceleration.
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


def check_gpu_availability():
    """Check and display GPU information."""
    print("🔧 GPU Information:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"     Memory: {memory_gb:.1f} GB")
        return True
    else:
        print("   No CUDA devices available")
        return False


def run_gpu_optimized_training():
    """Run GPU-optimized DeepLabV3+ training."""
    
    print("="*70)
    print("🚀 GPU-OPTIMIZED DEEPLABV3+ TRAINING")
    print("="*70)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    if gpu_available:
        # GPU configuration - optimized for performance
        train_config = {
            'model': '/workspaces/off-road-terrrain-seg-benchmark/src/cfg/models/deeplabv3plus_resnet50.yaml',
            'data': 'coco8-seg.yaml',
            'epochs': 10,      # More epochs for GPU
            'batch': 16,       # Larger batch size for GPU
            'imgsz': 640,      # Full resolution for better results
            'device': 'cuda',  # Use GPU
            'workers': 8,      # More workers for faster data loading
            'project': f'{project_root}/runs/gpu_training',
            'name': 'deeplabv3plus_gpu_optimized',
            'exist_ok': True,
            'verbose': True,
            'save': True,
            'plots': True,     # Enable plots for analysis
            'val': True,       # Enable validation
            'lr0': 0.01,       # Higher learning rate for larger batch
            'optimizer': 'Adam',
            'amp': True,       # Enable Automatic Mixed Precision for speed
            'cache': True,     # Cache images for faster loading
        }
        print("🚀 Using GPU-optimized configuration")
    else:
        # Fallback CPU configuration
        train_config = {
            'model': '/workspaces/off-road-terrrain-seg-benchmark/src/cfg/models/deeplabv3plus_resnet50.yaml',
            'data': 'coco8-seg.yaml',
            'epochs': 3,       # Fewer epochs for CPU
            'batch': 4,        # Smaller batch size for CPU
            'imgsz': 320,      # Medium resolution for CPU
            'device': 'cpu',   # Use CPU
            'workers': 2,      # Fewer workers for CPU
            'project': f'{project_root}/runs/cpu_training',
            'name': 'deeplabv3plus_cpu_fallback',
            'exist_ok': True,
            'verbose': True,
            'save': True,
            'plots': False,    # Disable plots for CPU to save time
            'val': True,
            'lr0': 0.001,      # Lower learning rate for smaller batch
            'optimizer': 'Adam',
            'amp': False,      # Disable AMP for CPU
            'cache': False,    # Disable cache for CPU
        }
        print("💻 Using CPU fallback configuration")
    
    print("\n📋 Training Configuration:")
    for key, value in train_config.items():
        print(f"   {key}: {value}")
    
    try:
        print("\n📦 Creating trainer...")
        trainer = DeepLabV3PlusSemanticSegmentationTrainer(overrides=train_config)
        
        print("🏗️  Setting up model...")
        model = trainer.get_model(cfg=train_config['model'])
        print(f"✅ Model created: {type(model).__name__}")
        
        if gpu_available:
            print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            if hasattr(model, 'model'):
                print(f"📊 Model size: ~{sum(p.numel() * 4 for p in model.parameters()) / 1024**2:.1f} MB")
        
        print(f"\n🚀 Starting {'GPU' if gpu_available else 'CPU'} training...")
        if gpu_available:
            print("   Using larger batch size and full resolution for better results")
        else:
            print("   Using smaller batch size and reduced resolution for CPU compatibility")
        
        # Start training
        results = trainer.train()
        
        print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Results saved to: {trainer.save_dir}")
        
        if results:
            print("📈 Training Results:")
            if hasattr(results, 'maps50'):
                print(f"   mAP@0.5: {results.maps50:.4f}")
            if hasattr(results, 'fitness'):
                print(f"   Fitness: {results.fitness:.4f}")
        
        device_type = "GPU" if gpu_available else "CPU"
        print(f"\n🎉 DeepLabV3+ {device_type} training pipeline is fully functional!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        print("\n📋 Full error traceback:")
        traceback.print_exc()
        return False


def main():
    """Main function to run GPU-optimized training."""
    
    print("🎯 DeepLabV3+ GPU-Optimized Training Test")
    print("   This script automatically detects GPU and uses optimal settings")
    
    # Run training
    success = run_gpu_optimized_training()
    
    if success:
        print("\n" + "="*70)
        print("🎊 SUCCESS! DeepLabV3+ training pipeline works perfectly!")
        print("="*70)
        print("\n📋 PRODUCTION RECOMMENDATIONS:")
        if torch.cuda.is_available():
            print("✅ GPU AVAILABLE - Recommended settings:")
            print("   • Epochs: 50-100 for full training")
            print("   • Batch size: 16-32 (depending on GPU memory)")
            print("   • Image size: 640-1024 for best results")
            print("   • Use AMP for faster training")
            print("   • Enable image caching for speed")
        else:
            print("💻 CPU ONLY - For production:")
            print("   • Consider using GPU for faster training")
            print("   • Reduce batch size if memory issues occur")
            print("   • Use smaller image sizes (320-480)")
        print("   • Use your custom off-road dataset")
        print("   • Fine-tune hyperparameters for your specific use case")
        
    else:
        print("\n" + "="*70)
        print("⚠️  Training test failed - check errors above")
        print("="*70)
    
    return success


if __name__ == "__main__":
    main()
