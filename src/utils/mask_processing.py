from typing import Dict, Any
import torch


def convert_instance_masks_to_semantic(batch: Dict[str, Any], ignore_index: int = 255) -> torch.Tensor:
    """
    Convert instance segmentation masks to semantic segmentation format.
    
    Transforms masks from YOLO instance format to semantic segmentation format suitable
    for pixel-wise classification losses and metrics:
    - Background pixels (0) → ignore_index (255) 
    - Object instance indices (1,2,...) → corresponding class labels from cls tensor
    
    Args:
        batch (Dict[str, Any]): Batch containing:
            - 'masks': List of 2D torch.Tensor instance masks (uint8/int)
            - 'cls': 1D tensor of class labels for each instance
            - 'batch_idx': 1D tensor mapping each instance to its batch item
        ignore_index (int): Value for background/ignored pixels. Defaults to 255.
        
    Returns:
        torch.Tensor: Semantic masks with shape [B, H, W] containing class indices.
        
    Examples:
        >>> batch = {
        ...     "masks": [torch.tensor([[0, 1, 2], [0, 1, 0]])],  # instances 1,2
        ...     "cls": torch.tensor([3, 7]),  # person, truck classes
        ...     "batch_idx": torch.tensor([0, 0])  # both in batch item 0
        ... }
        >>> semantic_masks = convert_instance_masks_to_semantic(batch)
        >>> # Result: [[255, 3, 7], [255, 3, 255]]  # bg=255, person=3, truck=7
        
    Notes:
        This function is essential for converting YOLO's instance segmentation format
        to the semantic segmentation format required by cross-entropy loss and mIoU metrics.
    """
    masks = batch["masks"]
    cls = batch["cls"]
    batch_idx = batch["batch_idx"]
    
    if len(masks) == 0:
        # Handle empty batch case - return a small tensor with background only
        return torch.full((1, 1, 1), ignore_index, dtype=torch.uint8)
    
    # Check if all masks are empty (no instances)
    all_empty = all(len(cls[batch_idx == i]) == 0 for i in range(len(masks)))
    if all_empty:
        # For empty masks, just convert all background to ignore_index
        processed = []
        max_h = max(mask.shape[-2] for mask in masks)
        max_w = max(mask.shape[-1] for mask in masks)
        
        for mask in masks:
            # Clone and pad if needed
            mask_proc = mask.clone()
            if mask_proc.shape != (max_h, max_w):
                padded_mask = torch.full((max_h, max_w), 0, dtype=mask_proc.dtype, device=mask_proc.device)
                h, w = mask_proc.shape
                padded_mask[:h, :w] = mask_proc
                mask_proc = padded_mask
            
            # Convert all to ignore_index (all background)
            semantic_mask = torch.full_like(mask_proc, ignore_index)
            processed.append(semantic_mask)
        
        return torch.stack(processed, dim=0)
    
    # Get the maximum dimensions to ensure consistent sizing
    max_h = max(mask.shape[-2] for mask in masks)
    max_w = max(mask.shape[-1] for mask in masks)
    
    processed = []

    for mask_idx, mask in enumerate(masks):
        # Clone to avoid modifying original data
        mask_proc = mask.clone()
        
        # Pad mask to maximum dimensions if needed
        if mask_proc.shape != (max_h, max_w):
            padded_mask = torch.full((max_h, max_w), 0, dtype=mask_proc.dtype, device=mask_proc.device)
            h, w = mask_proc.shape
            padded_mask[:h, :w] = mask_proc
            mask_proc = padded_mask
        
        # Get class labels for instances in this mask (batch index matches current mask index)
        mask_instances = cls[batch_idx == mask_idx]
        
        # Find max instance ID BEFORE converting background to ignore_index
        max_instance_id = int(mask_proc.max().item()) if mask_proc.numel() > 0 else 0
        
        # Convert background pixels to ignore index
        mask_proc[mask_proc == 0] = ignore_index
        
        # Create mapping from instance indices to class labels
        if max_instance_id > 0:
            # Create a mapping that can handle the ignore_index
            id_to_class = torch.full(
                (256,),  # Large enough to handle ignore_index (255)
                ignore_index, 
                dtype=mask_proc.dtype, 
                device=mask_proc.device
            )
            
            # Get unique instance IDs from the original mask (excluding background)
            unique_instances = torch.unique(mask)
            unique_instances = unique_instances[unique_instances != 0]
            unique_instances = torch.sort(unique_instances)[0]  # Sort to ensure consistent ordering
            
            # Map each unique instance ID to its corresponding class
            for i, instance_id in enumerate(unique_instances):
                if i < len(mask_instances):  # Bounds check
                    class_val = mask_instances[i]
                    if isinstance(class_val, torch.Tensor):
                        class_val = class_val.item()
                    id_to_class[instance_id.item()] = class_val
            
            # Apply the mapping vectorized
            semantic_mask = id_to_class[mask_proc.long()]
        else:
            # No instances, just background
            semantic_mask = mask_proc
            
        processed.append(semantic_mask.type_as(mask))

    return torch.stack(processed, dim=0)  # [B, H, W]