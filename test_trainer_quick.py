#!/usr/bin/env python3
"""
Simple test to check if the trainer works correctly.
"""

def test_trainer_quick():
    """Quick test for trainer functionality."""
    print("🧪 Testing DeepLabV3Plus Trainer Implementation...")
    
    try:
        # Test import
        from src.models.deeplabv3plus.train import DeepLabV3PlusSemanticSegmentationTrainer
        print("✅ Import successful")
        
        # Test basic initialization (should fail gracefully without data)
        try:
            trainer = DeepLabV3PlusSemanticSegmentationTrainer()
            print("❌ Should fail without data config")
        except Exception as e:
            if "data" in str(e).lower() or "none" in str(e).lower():
                print(f"✅ Expected error without data: {str(e)[:80]}...")
            else:
                print(f"❌ Unexpected error: {e}")
                
        # Test with minimal config
        overrides = {
            'data': 'coco8-seg.yaml',
            'model': 'deeplabv3plus_resnet50.yaml',  # This was missing!
            'epochs': 1,
            'batch': 1
        }
        
        try:
            trainer = DeepLabV3PlusSemanticSegmentationTrainer(overrides=overrides)
            print("✅ Trainer initialization with data config successful!")
            
            # Check attributes
            if hasattr(trainer, 'loss_names'):
                print(f"✅ loss_names: {trainer.loss_names}")
            else:
                print("❌ loss_names missing")
                
            # Check required methods
            required_methods = ['get_model', 'get_validator', 'progress_string', 'label_loss_items']
            for method in required_methods:
                if hasattr(trainer, method):
                    print(f"✅ Has {method}")
                else:
                    print(f"❌ Missing {method}")
                    
        except Exception as e:
            print(f"❌ Trainer initialization failed: {e}")
            import traceback
            traceback.print_exc()
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
        
    print("🎉 Trainer test completed!")
    return True

if __name__ == "__main__":
    test_trainer_quick()
