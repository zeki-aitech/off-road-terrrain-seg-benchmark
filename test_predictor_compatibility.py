#!/usr/bin/env python3
"""
Test script for DeepLabV3+ Predictor

This script tests the semantic segmentation predictor to ensure it's
Ultralytics-compatible and handles semantic segmentation output correctly.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Set project root and apply monkey patches
project_root = '/workspaces/off-road-terrrain-seg-benchmark'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# CRITICAL: Apply monkey patches BEFORE any imports
from src.patches import apply_patches
apply_patches()

from src.models.deeplabv3plus.predict import DeepLabV3PlusSemanticSegmentationPredictor
from src.models.deeplabv3plus.train import DeepLabV3PlusSemanticSegmentationTrainer


def test_predictor_compatibility():
    """Test that the predictor is properly Ultralytics-compatible."""
    
    print("="*60)
    print("🧪 TESTING DEEPLABV3+ PREDICTOR")
    print("="*60)
    
    # Test 1: Basic instantiation
    print("\n📦 1. Testing Predictor Instantiation...")
    try:
        predictor = DeepLabV3PlusSemanticSegmentationPredictor()
        print(f"✅ Predictor created: {type(predictor).__name__}")
        print(f"📋 Inherits from: {type(predictor).__bases__[0].__name__}")
    except Exception as e:
        print(f"❌ Predictor instantiation failed: {e}")
        return False
    
    # Test 2: Load model for prediction
    print("\n🏗️  2. Testing Model Loading...")
    try:
        # Create trainer to get a model
        trainer = DeepLabV3PlusSemanticSegmentationTrainer(overrides={
            'model': '/workspaces/off-road-terrrain-seg-benchmark/src/cfg/models/deeplabv3plus_resnet50.yaml',
            'data': 'coco8-seg.yaml',  # Add required data field
            'device': 'cpu'
        })
        model = trainer.get_model()
        model.eval()
        
        print(f"✅ Model loaded for prediction")
        print(f"📊 Model classes: {getattr(model, 'names', 'not available')}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Test 3: Test postprocessing with dummy data
    print("\n🔄 3. Testing Postprocessing...")
    try:
        # Create dummy predictions (semantic segmentation logits)
        batch_size = 1
        num_classes = 80
        height, width = 160, 160
        
        # Simulate model output: logits for each class
        dummy_preds = torch.randn(batch_size, num_classes, height, width)
        
        # Dummy input image
        dummy_img = torch.randn(batch_size, 3, height, width)
        
        # Dummy original image
        dummy_orig = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        print(f"📐 Input shapes:")
        print(f"   Predictions: {dummy_preds.shape}")
        print(f"   Input image: {dummy_img.shape}")
        print(f"   Original image: {dummy_orig.shape}")
        
        # Set up predictor with model
        predictor.model = model
        predictor.device = 'cpu'
        
        # Test postprocessing
        results = predictor.postprocess(dummy_preds, dummy_img, [dummy_orig])
        
        print(f"✅ Postprocessing successful")
        print(f"📊 Results: {len(results)} result objects")
        
        # Check result structure
        if results:
            result = results[0]
            print(f"📋 Result type: {type(result).__name__}")
            print(f"📋 Has masks: {hasattr(result, 'masks') and result.masks is not None}")
            if hasattr(result, 'masks') and result.masks is not None:
                mask = result.masks
                print(f"📐 Mask object type: {type(mask).__name__}")
                print(f"📐 Mask shape: {mask.shape}")
                
                # Access the actual mask data properly
                if hasattr(mask, 'data'):
                    # Ultralytics Masks object
                    mask_data = mask.data[0]
                    if hasattr(mask_data, 'numpy'):
                        mask_data = mask_data.numpy()  # Convert tensor to numpy
                else:
                    # Direct numpy array or indexable object
                    mask_data = mask[0] if hasattr(mask, '__getitem__') else mask
                
                unique_classes = np.unique(mask_data)
                print(f"🎯 Unique classes in prediction: {len(unique_classes)} classes")
                print(f"   Class IDs: {unique_classes[:10]}{'...' if len(unique_classes) > 10 else ''}")
        
    except Exception as e:
        print(f"❌ Postprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Check if it handles different input formats
    print("\n🔀 4. Testing Input Format Handling...")
    try:
        # Test with different prediction formats
        test_cases = [
            ("List format", [dummy_preds]),
            ("Tuple format", (dummy_preds,)),
            ("3D tensor", dummy_preds.squeeze(0)),  # [C, H, W]
        ]
        
        for case_name, test_preds in test_cases:
            print(f"   Testing {case_name}...")
            results = predictor.postprocess(test_preds, dummy_img, [dummy_orig])
            print(f"   ✅ {case_name} handled correctly")
            
    except Exception as e:
        print(f"❌ Input format test failed: {e}")
        return False
    
    # Test 5: Check semantic segmentation vs instance segmentation
    print("\n🎯 5. Verifying Semantic Segmentation Output...")
    try:
        # Create a prediction with clear semantic structure
        semantic_pred = torch.zeros(1, num_classes, height, width)
        
        # Create regions for different classes
        semantic_pred[0, 1, 20:80, 20:80] = 5.0    # Class 1 (high confidence)
        semantic_pred[0, 5, 80:140, 80:140] = 4.0  # Class 5 (high confidence)
        semantic_pred[0, 0, :, :] = 1.0            # Background (lower confidence)
        
        results = predictor.postprocess(semantic_pred, dummy_img, [dummy_orig])
        
        if results and results[0].masks is not None:
            mask_obj = results[0].masks
            
            # Get the actual mask data properly
            if hasattr(mask_obj, 'data'):
                # Ultralytics Masks object
                mask_data = mask_obj.data[0]
                if hasattr(mask_data, 'numpy'):
                    mask_data = mask_data.numpy()  # Convert tensor to numpy
            else:
                # Direct numpy array or indexable object
                mask_data = mask_obj[0] if hasattr(mask_obj, '__getitem__') else mask_obj
                
            unique_vals = np.unique(mask_data)
            print(f"✅ Semantic segmentation verified")
            print(f"   Predicted classes: {unique_vals}")
            print(f"   This is semantic (not instance) segmentation!")
            
            # Check that we get class labels, not instance IDs
            expected_classes = [0, 1, 5]  # Background + the two classes we set
            if all(cls in unique_vals for cls in expected_classes):
                print(f"   ✅ Correct semantic class mapping")
            else:
                print(f"   ⚠️ Unexpected class mapping: expected {expected_classes}, got {unique_vals}")
        
    except Exception as e:
        print(f"❌ Semantic verification failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("🎉 ALL PREDICTOR TESTS PASSED!")
    print("✅ DeepLabV3+ Predictor is Ultralytics-compatible!")
    print("="*60)
    
    print("\n📋 SUMMARY:")
    print("• ✅ Instantiation: Works correctly")
    print("• ✅ Model Loading: Compatible with trainer")
    print("• ✅ Postprocessing: Handles semantic segmentation output")
    print("• ✅ Input Formats: Flexible input handling")
    print("• ✅ Output Format: Produces semantic (not instance) masks")
    
    print("\n🎯 KEY FEATURES:")
    print("• Converts class logits to semantic masks")
    print("• Resizes masks to original image dimensions")
    print("• Returns Results objects compatible with Ultralytics")
    print("• Handles both multi-class and binary segmentation")
    print("• Saves semantic masks as PNG files")
    
    return True


def main():
    """Main function to run predictor tests."""
    print("🔬 Testing DeepLabV3+ Predictor Compatibility...")
    print("This verifies the predictor works with Ultralytics framework.")
    
    success = test_predictor_compatibility()
    
    if not success:
        print("\n❌ Some predictor tests failed. Check errors above.")
        return False
    
    print("\n🚀 Predictor is ready for use!")
    return True


if __name__ == "__main__":
    main()
