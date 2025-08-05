#!/usr/bin/env python3

import torch
import sys
sys.path.append('.')
from typing import Dict, Any

def convert_instance_masks_to_semantic_debug(batch: Dict[str, Any], ignore_index: int = 255) -> torch.Tensor:
    """Debug version of the mask processing function."""
    masks = batch["masks"]
    cls = batch["cls"]
    batch_idx = batch["batch_idx"]
    
    if len(masks) == 0:
        return torch.full((1, 1, 1), ignore_index, dtype=torch.uint8)
    
    all_empty = all(len(cls[batch_idx == i]) == 0 for i in range(len(masks)))
    if all_empty:
        max_h = max(mask.shape[-2] for mask in masks)
        max_w = max(mask.shape[-1] for mask in masks)
        result_masks = []
        for _ in masks:
            mask = torch.full((max_h, max_w), ignore_index, dtype=torch.uint8)
            mask[0, 0] = 0
            result_masks.append(mask)
        return torch.stack(result_masks, dim=0)
    
    max_h = max(mask.shape[-2] for mask in masks)
    max_w = max(mask.shape[-1] for mask in masks)
    
    processed = []

    for mask_idx, mask in enumerate(masks):
        print(f"\n--- Processing mask {mask_idx} ---")
        print(f"Original mask: {mask}")
        
        mask_proc = mask.clone()
        
        if mask_proc.shape != (max_h, max_w):
            padded_mask = torch.full((max_h, max_w), 0, dtype=mask_proc.dtype, device=mask_proc.device)
            h, w = mask_proc.shape
            padded_mask[:h, :w] = mask_proc
            mask_proc = padded_mask
            print(f"Padded mask: {mask_proc}")
        
        # Get class labels for instances in this mask (batch index matches current mask index)
        mask_instances = cls[batch_idx == mask_idx]
        print(f"mask_instances for mask {mask_idx}: {mask_instances}")
        
        # Find max instance ID BEFORE converting background to ignore_index
        max_instance_id = int(mask_proc.max().item()) if mask_proc.numel() > 0 else 0
        print(f"max_instance_id (before bg conversion): {max_instance_id}")
        
        # Convert background pixels to ignore index
        mask_proc[mask_proc == 0] = ignore_index
        print(f"After background->ignore: {mask_proc}")
        
        if max_instance_id > 0:
            # Create a mapping that can handle the ignore_index
            id_to_class = torch.full(
                (256,),  # Large enough to handle ignore_index (255)
                ignore_index, 
                dtype=mask_proc.dtype, 
                device=mask_proc.device
            )
            print(f"Initial id_to_class (first 10): {id_to_class[:10]}")
            
            # Get unique instance IDs from the original mask (excluding background)
            unique_instances = torch.unique(mask)
            unique_instances = unique_instances[unique_instances != 0]
            unique_instances = torch.sort(unique_instances)[0]
            print(f"unique_instances: {unique_instances}")
            
            for i, instance_id in enumerate(unique_instances):
                if i < len(mask_instances):
                    print(f"  Mapping instance_id {instance_id} -> class {mask_instances[i]}")
                    id_to_class[instance_id] = mask_instances[i].item()
            
            print(f"Final id_to_class[0:10]: {id_to_class[:10]}")
            print(f"id_to_class[255]: {id_to_class[255]}")
            
            semantic_mask = id_to_class[mask_proc.long()]
            print(f"semantic_mask after mapping: {semantic_mask}")
        else:
            semantic_mask = mask_proc
            print(f"No instances, using mask_proc: {semantic_mask}")
            
        processed.append(semantic_mask.type_as(mask))

    return torch.stack(processed, dim=0)

def test_mixed_batch():
    print("=== Testing Mixed Batch Indices with Debug ===")
    
    mask1 = torch.tensor([[0, 1, 2]], dtype=torch.uint8)
    mask2 = torch.tensor([[0, 3]], dtype=torch.uint8)
    
    batch = {
        "masks": [mask1, mask2],
        "cls": torch.tensor([5, 7, 12, 8]),
        "batch_idx": torch.tensor([0, 1, 0, 1])
    }
    
    print("Input:")
    print(f"mask1: {mask1}")
    print(f"mask2: {mask2}")
    print(f"cls: {batch['cls']}")
    print(f"batch_idx: {batch['batch_idx']}")
    
    result = convert_instance_masks_to_semantic_debug(batch)
    
    print(f"\nFinal result:")
    print(f"result[0]: {result[0]}")
    print(f"result[1]: {result[1]}")

if __name__ == "__main__":
    test_mixed_batch()
