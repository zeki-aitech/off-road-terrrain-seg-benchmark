import pytest
import torch
import torch.nn as nn

from src.utils.loss import DeepLabV3PlusSemanticSegmentationLoss


class MockModel(nn.Module):
    """Mock model for testing following Ultralytics patterns."""
    def __init__(self, device='cpu'):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, 1).to(device)

    def forward(self, x):
        return self.conv(x)


class TestDeepLabV3PlusLoss:
    """Test cases for DeepLabV3+ loss function following ClassificationValidator patterns."""

    def test_loss_initialization(self):
        """Test loss function initialization with device detection."""
        model = MockModel()
        loss_fn = DeepLabV3PlusSemanticSegmentationLoss(model)
        
        assert loss_fn.device == torch.device('cpu')

    def test_loss_forward_basic(self, sample_batch):
        """Test basic loss computation with realistic data."""
        model = MockModel()
        loss_fn = DeepLabV3PlusSemanticSegmentationLoss(model)
        
        # Create test predictions
        batch_size, num_classes, height, width = 2, 8, 64, 64
        preds = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
        
        # Compute loss
        loss, loss_detached = loss_fn(preds, sample_batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert isinstance(loss_detached, torch.Tensor)
        assert not loss_detached.requires_grad
        assert loss.item() >= 0  # Loss should be non-negative

    def test_loss_shape_validation(self):
        """Test that shape mismatches raise appropriate errors."""
        model = MockModel()
        loss_fn = DeepLabV3PlusSemanticSegmentationLoss(model)
        
        # Different batch sizes
        preds = torch.randn(2, 5, 64, 64)
        mask = torch.randint(0, 3, (64, 64), dtype=torch.uint8)
        batch = {
            "masks": [mask],  # Only one mask for batch size 2
            "cls": torch.tensor([1]),
            "batch_idx": torch.tensor([0])
        }
        
        with pytest.raises(ValueError, match="Batch size mismatch"):
            loss_fn(preds, batch)

    def test_loss_spatial_resize(self):
        """Test loss handles different spatial dimensions."""
        model = MockModel()
        loss_fn = DeepLabV3PlusSemanticSegmentationLoss(model)
        
        # Predictions and masks with different spatial dimensions
        preds = torch.randn(1, 3, 128, 128)
        mask = torch.randint(0, 3, (64, 64), dtype=torch.uint8)
        batch = {
            "masks": [mask],
            "cls": torch.tensor([1, 2]),
            "batch_idx": torch.tensor([0, 0])
        }
        
        # Should not raise error due to automatic resizing
        loss, _ = loss_fn(preds, batch)
        assert isinstance(loss, torch.Tensor)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_loss_cuda_compatibility(self):
        """Test loss function with CUDA tensors."""
        device = torch.device('cuda')
        model = MockModel(device=device)
        loss_fn = DeepLabV3PlusSemanticSegmentationLoss(model)
        
        preds = torch.randn(1, 3, 32, 32, device=device)
        mask = torch.randint(0, 3, (32, 32), dtype=torch.uint8)
        batch = {
            "masks": [mask],
            "cls": torch.tensor([1]),
            "batch_idx": torch.tensor([0])
        }
        
        loss, _ = loss_fn(preds, batch)
        assert loss.device == device

    def test_loss_ignore_index_handling(self):
        """Test that ignore index (255) is properly handled."""
        model = MockModel()
        loss_fn = DeepLabV3PlusSemanticSegmentationLoss(model)
        
        # Create mask with background (becomes ignore index 255)
        preds = torch.randn(1, 3, 4, 4)
        mask = torch.tensor([[0, 1], [0, 0]], dtype=torch.uint8)  # Mostly background
        batch = {
            "masks": [mask],
            "cls": torch.tensor([1]),
            "batch_idx": torch.tensor([0])
        }
        
        loss, _ = loss_fn(preds, batch)
        assert torch.isfinite(loss)  # Should handle ignore index gracefully

    def test_loss_nan_detection(self):
        """Test NaN detection and error raising."""
        model = MockModel()
        loss_fn = DeepLabV3PlusSemanticSegmentationLoss(model)
        
        # Create problematic inputs that might cause NaN
        preds = torch.full((1, 3, 2, 2), float('inf'))  # Inf values
        mask = torch.randint(0, 3, (2, 2), dtype=torch.uint8)
        batch = {
            "masks": [mask],
            "cls": torch.tensor([1]),
            "batch_idx": torch.tensor([0])
        }
        
        with pytest.raises(ValueError, match="Loss is NaN"):
            loss_fn(preds, batch)

    def test_loss_empty_batch(self, sample_empty_batch):
        """Test loss computation with empty batch (no instances)."""
        model = MockModel()
        loss_fn = DeepLabV3PlusSemanticSegmentationLoss(model)
        
        preds = torch.randn(1, 3, 32, 32)
        
        loss, _ = loss_fn(preds, sample_empty_batch)
        assert torch.isfinite(loss)