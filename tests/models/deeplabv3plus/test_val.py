import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch

from src.models.deeplabv3plus.val import DeepLabV3PlusSemanticSegmentationValidator


class TestDeepLabV3PlusValidator:
    """Test cases for DeepLabV3+ validator following ClassificationValidator patterns."""

    def test_validator_initialization(self, mock_args):
        """Test validator initialization."""
        validator = DeepLabV3PlusSemanticSegmentationValidator(args=mock_args)
        
        assert validator.args.task == "segment"
        assert hasattr(validator, 'metrics')
        assert validator.targets is None
        assert validator.pred is None

    def test_init_metrics(self, mock_model):
        """Test metrics initialization following ClassificationValidator pattern."""
        validator = DeepLabV3PlusSemanticSegmentationValidator()
        
        validator.init_metrics(mock_model)
        
        assert validator.names == mock_model.names
        assert validator.nc == len(mock_model.names)
        assert validator.pred == []
        assert validator.targets == []
        assert validator.metrics.nc == validator.nc

    def test_get_desc(self):
        """Test description string format."""
        validator = DeepLabV3PlusSemanticSegmentationValidator()
        desc = validator.get_desc()
        
        expected = ("%22s" + "%11s" * 3) % ("classes", "mIoU", "PixelAcc", "MeanAcc")
        assert desc == expected

    def test_preprocess(self, sample_batch, mock_args):
        """Test batch preprocessing."""
        validator = DeepLabV3PlusSemanticSegmentationValidator(args=mock_args)
        validator.device = torch.device('cpu')
        
        processed_batch = validator.preprocess(sample_batch)
        
        assert "semantic_masks" in processed_batch
        assert processed_batch["semantic_masks"].device == validator.device
        assert processed_batch["img"].dtype == torch.float32

    def test_postprocess_logits(self):
        """Test postprocessing of model logits."""
        validator = DeepLabV3PlusSemanticSegmentationValidator()
        
        # Test with 4D logits
        logits = torch.randn(2, 5, 32, 32)
        processed = validator.postprocess(logits)
        
        assert processed.shape == (2, 32, 32)  # Should apply argmax
        assert processed.dtype in [torch.long, torch.int64]

    def test_postprocess_list_input(self):
        """Test postprocessing with list/tuple input."""
        validator = DeepLabV3PlusSemanticSegmentationValidator()
        
        # Test with list input (common in multi-output models)
        list_input = [torch.randn(2, 5, 32, 32), torch.randn(2, 3, 32, 32)]
        processed = validator.postprocess(list_input)
        
        assert processed.shape == (2, 32, 32)

    def test_update_metrics(self):
        """Test metrics updating following ClassificationValidator pattern."""
        validator = DeepLabV3PlusSemanticSegmentationValidator()
        validator.pred = []
        validator.targets = []
        
        # Create test data
        preds = torch.randn(2, 5, 32, 32)  # Logits
        batch = {
            "semantic_masks": torch.randint(0, 5, (2, 32, 32), dtype=torch.long)
        }
        
        validator.update_metrics(preds, batch)
        
        assert len(validator.pred) == 1
        assert len(validator.targets) == 1
        assert validator.pred[0].shape == (2, 32, 32)  # Should be class indices
        assert validator.targets[0].shape == (2, 32, 32)

    def test_update_metrics_resize(self):
        """Test metrics updating with spatial size mismatch."""
        validator = DeepLabV3PlusSemanticSegmentationValidator()
        validator.pred = []
        validator.targets = []
        
        # Different spatial sizes
        preds = torch.randn(1, 3, 64, 64)
        batch = {
            "semantic_masks": torch.randint(0, 3, (1, 32, 32), dtype=torch.long)
        }
        
        validator.update_metrics(preds, batch)
        
        # Targets should be resized to match predictions
        assert validator.pred[0].shape == (1, 64, 64)
        assert validator.targets[0].shape == (1, 64, 64)

    def test_finalize_metrics(self):
        """Test metrics finalization following ClassificationValidator pattern."""
        validator = DeepLabV3PlusSemanticSegmentationValidator()
        validator.pred = [torch.randint(0, 3, (2, 32, 32))]
        validator.targets = [torch.randint(0, 3, (2, 32, 32))]
        validator.speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5}
        validator.save_dir = Path("/tmp/test")
        
        # Mock the metrics process method
        validator.metrics.process = Mock()
        
        validator.finalize_metrics()
        
        validator.metrics.process.assert_called_once_with(validator.targets, validator.pred)
        assert validator.metrics.speed == validator.speed
        assert validator.metrics.save_dir == validator.save_dir

    def test_get_stats(self, mock_metrics_results):
        """Test getting statistics."""
        validator = DeepLabV3PlusSemanticSegmentationValidator()
        
        # Set the underlying metric values instead of trying to set results_dict
        validator.metrics.miou = mock_metrics_results["mIoU"]
        validator.metrics.pixel_acc = mock_metrics_results["PixelAcc"]
        validator.metrics.mean_class_acc = mock_metrics_results["MeanAcc"]
        
        stats = validator.get_stats()
        
        # Check that the stats contain the expected values
        assert stats["metrics/mIoU"] == mock_metrics_results["mIoU"]
        assert stats["metrics/pixel_accuracy"] == mock_metrics_results["PixelAcc"]
        assert stats["metrics/mean_class_accuracy"] == mock_metrics_results["MeanAcc"]

    def test_print_results(self, mock_metrics_results):
        """Test results printing."""
        validator = DeepLabV3PlusSemanticSegmentationValidator()
        
        # Set the underlying metric values instead of trying to set results_dict
        validator.metrics.miou = mock_metrics_results["mIoU"]
        validator.metrics.pixel_acc = mock_metrics_results["PixelAcc"]
        validator.metrics.mean_class_acc = mock_metrics_results["MeanAcc"]
        
        # Test that print_results doesn't crash - it uses LOGGER.info so we can't easily capture output
        try:
            validator.print_results()
            # If we get here without exception, the test passes
            assert True
        except Exception as e:
            pytest.fail(f"print_results() raised an exception: {e}")

    def test_plot_methods_safe(self):
        """Test that plotting methods don't crash (graceful handling)."""
        validator = DeepLabV3PlusSemanticSegmentationValidator()
        validator.args = Mock()
        validator.args.plots = True
        
        batch = {"img": torch.randn(2, 3, 32, 32)}
        preds = torch.randint(0, 3, (2, 32, 32))
        
        # These should not raise exceptions
        validator.plot_val_samples(batch, 0)
        validator.plot_predictions(batch, preds, 0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_validator_cuda_compatibility(self, sample_batch, mock_args):
        """Test validator with CUDA tensors."""
        device = torch.device('cuda')
        validator = DeepLabV3PlusSemanticSegmentationValidator(args=mock_args)
        validator.device = device
        
        # Move batch to CUDA
        sample_batch["img"] = sample_batch["img"].to(device)
        
        processed_batch = validator.preprocess(sample_batch)
        assert processed_batch["img"].device == device
        assert processed_batch["semantic_masks"].device == device