#!/usr/bin/env python3
"""
Quick test to verify the DeepLabV3Plus trainer works correctly.
"""

def test_trainer_inheritance():
    """Test that the trainer inherits correctly from SegmentationTrainer."""
    from src.models.deeplabv3plus.train import DeepLabV3PlusSemanticSegmentationTrainer
    from ultralytics.models.yolo.segment import SegmentationTrainer
    
    print("=== Testing Trainer Inheritance ===")
    
    # Check inheritance
    assert issubclass(DeepLabV3PlusSemanticSegmentationTrainer, SegmentationTrainer)
    print("✅ Correctly inherits from SegmentationTrainer")
    
    # Check MRO
    mro = DeepLabV3PlusSemanticSegmentationTrainer.__mro__
    print(f"✅ MRO: {[cls.__name__ for cls in mro]}")
    
    # Check required methods exist
    required_methods = ['get_model', 'get_validator', 'progress_string', 'label_loss_items']
    for method in required_methods:
        assert hasattr(DeepLabV3PlusSemanticSegmentationTrainer, method)
        print(f"✅ Has method: {method}")
    
    print("✅ All trainer inheritance tests passed!")

def test_trainer_methods():
    """Test that trainer methods work correctly."""
    from src.models.deeplabv3plus.train import DeepLabV3PlusSemanticSegmentationTrainer
    
    print("\n=== Testing Trainer Methods ===")
    
    # Test progress string
    trainer = type('MockTrainer', (), {})()  # Mock trainer
    trainer.loss_names = ["loss"]
    
    # Bind the method to our mock trainer
    progress_method = DeepLabV3PlusSemanticSegmentationTrainer.progress_string
    result = progress_method(trainer)
    
    print(f"✅ Progress string: {result[:50]}...")
    assert "Epoch" in result
    assert "loss" in result
    
    # Test label_loss_items
    import torch
    loss_items = torch.tensor([0.5])
    label_method = DeepLabV3PlusSemanticSegmentationTrainer.label_loss_items
    result = label_method(trainer, loss_items, "train")
    
    print(f"✅ Label loss items: {result}")
    assert "train/loss" in result
    assert result["train/loss"] == 0.5
    
    print("✅ All trainer method tests passed!")

if __name__ == "__main__":
    test_trainer_inheritance()
    test_trainer_methods()
    print("\n🎉 ALL TESTS PASSED - Your trainer implementation is excellent!")
