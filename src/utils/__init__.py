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
    processed = []

    for mask_idx, mask in enumerate(masks):
        # Clone to avoid modifying original data
        mask_proc = mask.clone()
        
        # Convert background pixels to ignore index
        mask_proc[mask_proc == 0] = ignore_index
        
        # Get class labels for instances in this mask
        mask_instances = cls[batch_idx == mask_idx]
        
        # Create mapping from instance indices to class labels
        max_instance_id = int(mask_proc.max().item())
        id_to_class = torch.full(
            (max_instance_id + 1,), 
            ignore_index, 
            dtype=mask_proc.dtype, 
            device=mask_proc.device
        )
        
        # Map each instance ID to its corresponding class
        for instance_idx, class_label in enumerate(mask_instances, start=1):
            if instance_idx <= max_instance_id:  # Bounds check
                id_to_class[instance_idx] = class_label
        
        # Apply the mapping vectorized
        semantic_mask = id_to_class[mask_proc.long()]
        processed.append(semantic_mask.type_as(mask))

    return torch.stack(processed, dim=0)  # [B, H, W]