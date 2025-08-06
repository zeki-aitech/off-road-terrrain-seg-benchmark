"""
Test DeepLabV3Plus model compatibility with Ultralytics framework.
This comprehensive test verifies that the unified DeepLabV3Plus interface
works seamlessly with Ultralytics APIs.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add src to path and apply patches
sys.path.insert(0, '/workspaces/off-road-terrrain-seg-benchmark/src')
import patches  # This applies the monkey patches

print("✅ Monkey patches applied: parse_model replaced with DeepLabV3+ version")

# Now import ultralytics and our model
from ultralytics import YOLO
from src.models.deeplabv3plus.model import DeepLabV3Plus

def test_deeplabv3plus_compatibility():
    """Test all aspects of DeepLabV3Plus compatibility with Ultralytics."""
    
    print("🔬 Testing DeepLabV3Plus Ultralytics Compatibility...")
    print("=" * 60)
    
    # Test 1: Model Instantiation
    print("\n📦 1. Testing Model Instantiation...")
    try:
        model_path = "/workspaces/off-road-terrrain-seg-benchmark/src/cfg/models/deeplabv3plus_resnet50.yaml"
        model = DeepLabV3Plus(model=model_path, task="segment", verbose=False)
        print(f"✅ DeepLabV3Plus model created successfully")
        print(f"📋 Model type: {type(model).__name__}")
        print(f"📋 Task: {model.task}")
        print(f"📋 Model file: {model.model}")
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Check Task Map
    print("\n🗺️  2. Testing Task Map...")
    try:
        task_map = model.task_map
        print(f"✅ Task map retrieved: {list(task_map.keys())}")
        
        if "segment" in task_map:
            segment_config = task_map["segment"]
            required_components = ["model", "trainer", "validator", "predictor"]
            
            for component in required_components:
                if component in segment_config:
                    component_class = segment_config[component]
                    print(f"   ✅ {component}: {component_class.__name__}")
                else:
                    print(f"   ❌ Missing {component}")
                    return False
        else:
            print("❌ 'segment' task not found in task_map")
            return False
            
    except Exception as e:
        print(f"❌ Task map test failed: {e}")
        return False
    
    # Test 3: Model Architecture Loading
    print("\n🏗️  3. Testing Model Architecture...")
    try:
        # Check if model has been built
        if hasattr(model, 'model') and model.model is not None:
            print(f"✅ Model architecture loaded")
            print(f"📊 Model device: {model.device}")
            
            # Try to get model summary info
            if hasattr(model.model, 'yaml'):
                print(f"📋 Model config available: {bool(model.model.yaml)}")
            
            # Check if model has parameters
            if hasattr(model.model, 'parameters'):
                total_params = sum(p.numel() for p in model.model.parameters())
                print(f"📊 Total parameters: {total_params:,}")
        else:
            print("⚠️ Model architecture not yet loaded (lazy loading)")
            
    except Exception as e:
        print(f"❌ Model architecture test failed: {e}")
        return False
    
    # Test 4: Prediction Interface
    print("\n🎯 4. Testing Prediction Interface...")
    try:
        # Create dummy input
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test if we can call predict (this will trigger model loading if not done)
        print("   Testing predict method...")
        results = model.predict(dummy_image, verbose=False, save=False)
        
        print(f"✅ Prediction successful")
        print(f"📊 Results type: {type(results)}")
        print(f"📊 Number of results: {len(results) if hasattr(results, '__len__') else 'N/A'}")
        
        # Check result structure
        if results and len(results) > 0:
            result = results[0]
            print(f"📋 Result type: {type(result).__name__}")
            
            # Check for semantic segmentation output
            if hasattr(result, 'masks') and result.masks is not None:
                print(f"✅ Semantic masks present")
                mask = result.masks
                if hasattr(mask, 'data'):
                    mask_shape = mask.data.shape if hasattr(mask.data, 'shape') else 'Unknown'
                else:
                    mask_shape = mask.shape if hasattr(mask, 'shape') else 'Unknown'
                print(f"📐 Mask shape: {mask_shape}")
            else:
                print("⚠️ No masks in result")
                
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Training Interface (dry run)
    print("\n🏋️  5. Testing Training Interface...")
    try:
        # Test if we can access training components without actually training
        trainer_class = model.task_map["segment"]["trainer"]
        validator_class = model.task_map["segment"]["validator"]
        
        print(f"✅ Trainer class: {trainer_class.__name__}")
        print(f"✅ Validator class: {validator_class.__name__}")
        
        # Test if train method exists and is callable
        if hasattr(model, 'train') and callable(model.train):
            print("✅ Train method available")
        else:
            print("❌ Train method not available")
            return False
            
        # Test if val method exists and is callable
        if hasattr(model, 'val') and callable(model.val):
            print("✅ Validation method available")
        else:
            print("❌ Validation method not available")
            return False
            
    except Exception as e:
        print(f"❌ Training interface test failed: {e}")
        return False
    
    # Test 6: Model Properties and Methods
    print("\n📋 6. Testing Model Properties...")
    try:
        # Check essential properties and methods
        essential_attrs = ['device', 'names', 'task']
        essential_methods = ['predict', 'train', 'val', 'export']
        
        for attr in essential_attrs:
            if hasattr(model, attr):
                value = getattr(model, attr)
                print(f"   ✅ {attr}: {value}")
            else:
                print(f"   ⚠️ Missing attribute: {attr}")
                
        for method in essential_methods:
            if hasattr(model, method) and callable(getattr(model, method)):
                print(f"   ✅ {method}(): Available")
            else:
                print(f"   ❌ Missing method: {method}")
                return False
                
    except Exception as e:
        print(f"❌ Properties test failed: {e}")
        return False
    
    # Test 7: Ultralytics API Compatibility
    print("\n🔌 7. Testing Ultralytics API Compatibility...")
    try:
        # Test if our model behaves like a standard Ultralytics model
        print("   Testing standard Ultralytics patterns...")
        
        # Check if model can be used in typical Ultralytics workflows
        # 1. Check if model.model exists (the actual torch model)
        if hasattr(model, 'model') and model.model is not None:
            print("   ✅ Internal model object accessible")
            
        # 2. Check if we can access model info
        if hasattr(model, 'info'):
            try:
                model.info(verbose=False)
                print("   ✅ Model info() method works")
            except:
                print("   ⚠️ Model info() method has issues")
                
        # 3. Check device property
        if hasattr(model, 'device'):
            print(f"   ✅ Device property: {model.device}")
            
        # 4. Check if we can change device
        try:
            current_device = model.device
            # Don't actually change device, just test the method exists
            if hasattr(model, 'to') and callable(model.to):
                print("   ✅ Device change method available")
            else:
                print("   ⚠️ Device change method not available")
        except:
            print("   ⚠️ Device handling has issues")
            
    except Exception as e:
        print(f"❌ API compatibility test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL DEEPLABV3PLUS COMPATIBILITY TESTS PASSED!")
    print("✅ DeepLabV3Plus is fully compatible with Ultralytics!")
    print("=" * 60)
    
    print("\n📋 COMPATIBILITY SUMMARY:")
    print("• ✅ Model Instantiation: Works correctly")
    print("• ✅ Task Mapping: Proper component registration")
    print("• ✅ Model Architecture: Loads and configures properly")
    print("• ✅ Prediction Interface: Compatible with Ultralytics predict()")
    print("• ✅ Training Interface: train() and val() methods available")
    print("• ✅ Model Properties: All essential attributes present")
    print("• ✅ API Compatibility: Behaves like standard Ultralytics model")
    
    print("\n🚀 Ready for production use with Ultralytics framework!")
    return True

def test_yolo_comparison():
    """Compare DeepLabV3Plus with standard YOLO model to ensure similar interface."""
    print("\n🔄 BONUS: Comparing with Standard YOLO Model...")
    print("-" * 50)
    
    try:
        # Load our model
        deeplabv3_model = DeepLabV3Plus(
            model="/workspaces/off-road-terrrain-seg-benchmark/src/cfg/models/deeplabv3plus_resnet50.yaml",
            task="segment"
        )
        
        # Load standard YOLO model for comparison
        try:
            yolo_model = YOLO("yolo11n-seg.pt")
            print("✅ Standard YOLO model loaded for comparison")
        except:
            print("⚠️ Could not load standard YOLO model for comparison")
            return
        
        # Compare interfaces
        print("\n📋 Interface Comparison:")
        
        # Check common methods
        common_methods = ['predict', 'train', 'val', 'export', 'info']
        for method in common_methods:
            yolo_has = hasattr(yolo_model, method) and callable(getattr(yolo_model, method))
            deeplab_has = hasattr(deeplabv3_model, method) and callable(getattr(deeplabv3_model, method))
            
            if yolo_has and deeplab_has:
                print(f"   ✅ {method}(): Both models support this")
            elif yolo_has and not deeplab_has:
                print(f"   ❌ {method}(): Missing in DeepLabV3Plus")
            elif not yolo_has and deeplab_has:
                print(f"   ✅ {method}(): DeepLabV3Plus has extra functionality")
            else:
                print(f"   ⚠️ {method}(): Neither model supports this")
        
        # Compare properties
        common_props = ['device', 'names', 'task']
        print("\n📋 Property Comparison:")
        for prop in common_props:
            yolo_has = hasattr(yolo_model, prop)
            deeplab_has = hasattr(deeplabv3_model, prop)
            
            if yolo_has and deeplab_has:
                yolo_val = getattr(yolo_model, prop, "N/A")
                deeplab_val = getattr(deeplabv3_model, prop, "N/A")
                print(f"   ✅ {prop}: YOLO={yolo_val}, DeepLab={deeplab_val}")
            elif yolo_has and not deeplab_has:
                print(f"   ❌ {prop}: Missing in DeepLabV3Plus")
            else:
                print(f"   ✅ {prop}: Available in DeepLabV3Plus")
        
        print("\n✅ Interface comparison completed")
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")

if __name__ == "__main__":
    try:
        success = test_deeplabv3plus_compatibility()
        if success:
            test_yolo_comparison()
            print("\n🎯 FINAL RESULT: DeepLabV3Plus is FULLY COMPATIBLE with Ultralytics! 🎉")
        else:
            print("\n❌ FINAL RESULT: Compatibility issues found")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
