import pytest
import torch
import numpy as np

from src.utils.mask_processing import convert_instance_masks_to_semantic


class TestMaskProcessing:
    """Test cases for mask processing utilities following Ultralytics patterns."""

    def test_convert_instance_masks_to_semantic_basic(self):
        """Test basic instance to semantic mask conversion."""
        # Create test data following YOLO segmentation format
        mask = torch.tensor([[0, 1, 2], [0, 1, 0], [3, 0, 0]], dtype=torch.uint8)
        batch = {
            "masks": [mask],
            "cls": torch.tensor([5, 7, 10]),  # Classes for instances 1, 2, 3
            "batch_idx": torch.tensor([0, 0, 0])  # All in first batch item
        }
        
        # Convert masks
        result = convert_instance_masks_to_semantic(batch, ignore_index=255)
        
        # Expected: background(0)→255, instance1→class5, instance2→class7, instance3→class10
        expected = torch.tensor([[255, 5, 7], [255, 5, 255], [10, 255, 255]], dtype=torch.uint8)
        
        assert result.shape == (1, 3, 3), f"Expected shape (1, 3, 3), got {result.shape}"
        assert torch.equal(result[0], expected), f"Expected {expected}, got {result[0]}"

    def test_convert_instance_masks_multiple_batches(self):
        """Test conversion with multiple batch items (following BaseValidator pattern)."""
        mask1 = torch.tensor([[0, 1], [2, 0]], dtype=torch.uint8)
        mask2 = torch.tensor([[0, 1], [0, 1]], dtype=torch.uint8)
        
        batch = {
            "masks": [mask1, mask2],
            "cls": torch.tensor([3, 5, 8]),  # Classes
            "batch_idx": torch.tensor([0, 0, 1])  # First two instances in batch 0, last in batch 1
        }
        
        result = convert_instance_masks_to_semantic(batch)
        
        # Batch 0: instance1→class3, instance2→class5
        expected1 = torch.tensor([[255, 3], [5, 255]], dtype=torch.uint8)
        # Batch 1: instance1→class8
        expected2 = torch.tensor([[255, 8], [255, 8]], dtype=torch.uint8)
        
        assert result.shape == (2, 2, 2)
        assert torch.equal(result[0], expected1)
        assert torch.equal(result[1], expected2)

    def test_convert_instance_masks_empty_instances(self):
        """Test conversion with no instances (background only)."""
        mask = torch.zeros((2, 2), dtype=torch.uint8)
        batch = {
            "masks": [mask],
            "cls": torch.tensor([]),
            "batch_idx": torch.tensor([])
        }
        
        result = convert_instance_masks_to_semantic(batch)
        expected = torch.full((2, 2), 255, dtype=torch.uint8)
        
        assert torch.equal(result[0], expected)

    def test_convert_instance_masks_custom_ignore_index(self):
        """Test conversion with custom ignore index."""
        mask = torch.tensor([[0, 1]], dtype=torch.uint8)
        batch = {
            "masks": [mask],
            "cls": torch.tensor([42]),
            "batch_idx": torch.tensor([0])
        }
        
        result = convert_instance_masks_to_semantic(batch, ignore_index=99)
        expected = torch.tensor([[99, 42]], dtype=torch.uint8)
        
        assert torch.equal(result[0], expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_convert_instance_masks_device_preservation(self):
        """Test that device is preserved during conversion (following Ultralytics device handling)."""
        device = torch.device('cuda')
        mask = torch.tensor([[0, 1]], dtype=torch.uint8, device=device)
        batch = {
            "masks": [mask],
            "cls": torch.tensor([1], device=device),
            "batch_idx": torch.tensor([0], device=device)
        }
        
        result = convert_instance_masks_to_semantic(batch)
        assert result.device == device

    def test_convert_instance_masks_dtype_preservation(self, mask_dtype):
        """Test that mask dtype is preserved (parametrized test)."""
        mask = torch.tensor([[0, 1]], dtype=mask_dtype)
        batch = {
            "masks": [mask],
            "cls": torch.tensor([1]),
            "batch_idx": torch.tensor([0])
        }
        
        result = convert_instance_masks_to_semantic(batch)
        assert result.dtype == mask_dtype, f"Expected {mask_dtype}, got {result.dtype}"

    def test_convert_instance_masks_large_instance_ids(self):
        """Test conversion with large instance IDs."""
        mask = torch.tensor([[0, 1, 10, 50]], dtype=torch.uint8)
        batch = {
            "masks": [mask],
            "cls": torch.tensor([2, 7, 15]),  # Classes for instances 1, 10, 50
            "batch_idx": torch.tensor([0, 0, 0])
        }
        
        result = convert_instance_masks_to_semantic(batch)
        # All instances should be mapped since we have enough classes
        expected = torch.tensor([[255, 2, 7, 15]], dtype=torch.uint8)
        
        assert torch.equal(result[0], expected)

    def test_convert_instance_masks_realistic_segmentation(self, sample_batch):
        """Test with realistic segmentation batch from conftest."""
        result = convert_instance_masks_to_semantic(sample_batch)
        
        # Should have correct batch dimension
        assert result.shape[0] == len(sample_batch["masks"])
        
        # Background pixels should be 255
        assert (result == 255).sum() > 0  # Should have background pixels
        
        # Should contain class labels from cls tensor
        unique_classes = torch.unique(result)
        cls_values = sample_batch["cls"].unique()
        
        # All non-background values should be from cls or ignore_index
        for val in unique_classes:
            assert val.item() == 255 or val in cls_values

    def test_convert_instance_masks_bounds_checking(self):
        """Test bounds checking for instance IDs."""
        # Create mask with instance ID larger than cls array
        mask = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.uint8)
        batch = {
            "masks": [mask],
            "cls": torch.tensor([10, 20]),  # Only 2 classes for 5 instances
            "batch_idx": torch.tensor([0, 0])
        }
        
        result = convert_instance_masks_to_semantic(batch)
        
        # Only instances 1 and 2 should be mapped
        expected = torch.tensor([[255, 10, 20, 255, 255, 255]], dtype=torch.uint8)
        assert torch.equal(result[0], expected)

    def test_convert_instance_masks_mixed_batch_indices(self):
        """Test with mixed batch indices (following classification validator patterns)."""
        mask1 = torch.tensor([[0, 1, 2]], dtype=torch.uint8)
        mask2 = torch.tensor([[0, 3]], dtype=torch.uint8)
        
        batch = {
            "masks": [mask1, mask2],
            "cls": torch.tensor([5, 7, 12, 8]),  # 4 total instances
            "batch_idx": torch.tensor([0, 1, 0, 1])  # Mixed assignment
        }
        
        result = convert_instance_masks_to_semantic(batch)
        
        # Batch 0: instances with batch_idx==0 → cls[0], cls[2] = 5, 12
        expected1 = torch.tensor([[255, 5, 12]], dtype=torch.uint8)
        # Batch 1: instances with batch_idx==1 → cls[1], cls[3] = 7, 8
        # Note: mask2 gets padded to match max width, so last element is background (255)
        expected2 = torch.tensor([[255, 7, 255]], dtype=torch.uint8)
        
        assert torch.equal(result[0], expected1)
        assert torch.equal(result[1], expected2)

    def test_convert_instance_masks_edge_cases(self):
        """Test edge cases that might occur during validation."""
        # Single pixel mask
        mask = torch.tensor([[1]], dtype=torch.uint8)
        batch = {
            "masks": [mask],
            "cls": torch.tensor([9]),
            "batch_idx": torch.tensor([0])
        }
        
        result = convert_instance_masks_to_semantic(batch)
        assert result.shape == (1, 1, 1)
        assert result[0, 0, 0].item() == 9

    def test_convert_instance_masks_different_ignore_indices(self, ignore_index):
        """Test with different ignore index values (parametrized)."""
        mask = torch.tensor([[0, 1]], dtype=torch.uint8)
        batch = {
            "masks": [mask],
            "cls": torch.tensor([5]),
            "batch_idx": torch.tensor([0])
        }
        
        result = convert_instance_masks_to_semantic(batch, ignore_index=ignore_index)
        expected = torch.tensor([[ignore_index, 5]], dtype=torch.uint8)
        
        assert torch.equal(result[0], expected)

    def test_convert_instance_masks_performance(self, large_batch):
        """Test performance with larger batches (following Ultralytics performance testing)."""
        import time
        
        start_time = time.time()
        result = convert_instance_masks_to_semantic(large_batch)
        end_time = time.time()
        
        # Should complete in reasonable time (< 1 second for test data)
        assert end_time - start_time < 1.0
        
        # Result should have correct shape
        assert result.shape[0] == len(large_batch["masks"])

    def test_convert_instance_masks_memory_efficiency(self):
        """Test memory efficiency by checking no unnecessary copies are made."""
        mask = torch.tensor([[0, 1, 2]], dtype=torch.uint8)
        original_mask = mask.clone()
        
        batch = {
            "masks": [mask],
            "cls": torch.tensor([5, 7]),
            "batch_idx": torch.tensor([0, 0])
        }
        
        result = convert_instance_masks_to_semantic(batch)
        
        # Original mask should be unchanged
        assert torch.equal(mask, original_mask)

    def test_convert_instance_masks_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with empty masks list - should handle gracefully
        batch = {
            "masks": [],
            "cls": torch.tensor([]),
            "batch_idx": torch.tensor([])
        }
        
        result = convert_instance_masks_to_semantic(batch)
        # Should return a small tensor with ignore_index
        assert result.shape == (1, 1, 1)
        assert result.item() == 255

    def test_convert_instance_masks_coco_style(self, mock_coco_model):
        """Test with COCO-style class names (following Ultralytics COCO patterns)."""
        mask = torch.tensor([[0, 1, 2, 3]], dtype=torch.uint8)
        batch = {
            "masks": [mask],
            "cls": torch.tensor([0, 2, 7]),  # person, car, truck
            "batch_idx": torch.tensor([0, 0, 0])
        }
        
        result = convert_instance_masks_to_semantic(batch)
        
        # Should map to actual COCO class indices
        expected = torch.tensor([[255, 0, 2, 7]], dtype=torch.uint8)
        assert torch.equal(result[0], expected)

    def test_convert_instance_masks_integration_with_validator(self, sample_batch, mock_args):
        """Test integration with validator workflow (simulating actual usage)."""
        # Simulate validator preprocessing
        processed_masks = convert_instance_masks_to_semantic(sample_batch)
        
        # Should be ready for loss calculation
        assert processed_masks.dtype in [torch.uint8, torch.long, torch.int32]
        assert len(processed_masks.shape) == 3  # [B, H, W]
        
        # Background pixels should be ignore index
        assert (processed_masks == 255).any()