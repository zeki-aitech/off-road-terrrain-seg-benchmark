#!/usr/bin/env python3

import torch
import sys
sys.path.append('.')
from src.utils.mask_processing import convert_instance_masks_to_semantic

def test_mixed_batch():
    print("=== Testing Mixed Batch Indices ===")
    
    # Test case from the failing test
    mask1 = torch.tensor([[0, 1, 2]], dtype=torch.uint8)
    mask2 = torch.tensor([[0, 3]], dtype=torch.uint8)
    
    batch = {
        "masks": [mask1, mask2],
        "cls": torch.tensor([5, 7, 12, 8]),  # 4 total instances
        "batch_idx": torch.tensor([0, 1, 0, 1])  # Mixed assignment
    }
    
    print("Input:")
    print(f"mask1: {mask1}")
    print(f"mask2: {mask2}")
    print(f"cls: {batch['cls']}")
    print(f"batch_idx: {batch['batch_idx']}")
    
    # For mask1 (mask_idx=0), instances with batch_idx==0
    mask_instances_0 = batch['cls'][batch['batch_idx'] == 0]
    print(f"\nFor mask1 (mask_idx=0), instances with batch_idx==0: {mask_instances_0}")
    print(f"These should map to instance IDs [1, 2] in mask1")
    
    # For mask2 (mask_idx=1), instances with batch_idx==1  
    mask_instances_1 = batch['cls'][batch['batch_idx'] == 1]
    print(f"\nFor mask2 (mask_idx=1), instances with batch_idx==1: {mask_instances_1}")
    print(f"These should map to instance ID [3] in mask2")
    
    result = convert_instance_masks_to_semantic(batch)
    
    print(f"\nActual result:")
    print(f"result[0]: {result[0]}")
    print(f"result[1]: {result[1]}")
    
    print(f"\nExpected:")
    print(f"result[0] should be: [[255, 5, 12]]")  # bg=255, instance 1->class 5, instance 2->class 12
    print(f"result[1] should be: [[255, 7]]")      # bg=255, instance 3->class 7, but needs padding
    
    # Let me check the sizes
    print(f"\nSizes:")
    print(f"mask1 shape: {mask1.shape}")
    print(f"mask2 shape: {mask2.shape}")
    print(f"result[0] shape: {result[0].shape}")
    print(f"result[1] shape: {result[1].shape}")

if __name__ == "__main__":
    test_mixed_batch()
