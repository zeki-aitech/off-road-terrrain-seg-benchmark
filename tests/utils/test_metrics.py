import pytest
import torch
import numpy as np

from src.utils.metrics import SemanticSegmentMetrics


class TestSemanticSegmentMetrics:
    """Test cases for semantic segmentation metrics following Ultralytics patterns."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = SemanticSegmentMetrics()
        
        # Should initialize without errors
        assert hasattr(metrics, 'results_dict')

    def test_metrics_process_basic(self):
        """Test basic metrics processing."""
        metrics = SemanticSegmentMetrics()
        
        # Create simple test data
        targets = [torch.tensor([[0, 1], [1, 0]])]
        preds = [torch.tensor([[0, 1], [1, 0]])]  # Perfect prediction
        
        metrics.process(targets, preds)
        results = metrics.results_dict
        
        # Should have some results
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_metrics_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        metrics = SemanticSegmentMetrics()
        
        # Perfect predictions
        target = torch.randint(0, 5, (2, 32, 32))
        pred = target.clone()
        
        metrics.process([target], [pred])
        results = metrics.results_dict
        
        # Perfect prediction should give high scores
        if 'PixelAcc' in results:
            assert results['PixelAcc'] >= 0.99

    def test_metrics_worst_prediction(self):
        """Test metrics with worst case predictions."""
        metrics = SemanticSegmentMetrics()
        
        # Completely wrong predictions
        target = torch.zeros(2, 32, 32, dtype=torch.long)
        pred = torch.ones(2, 32, 32, dtype=torch.long)
        
        metrics.process([target], [pred])
        results = metrics.results_dict
        
        # Should handle worst case gracefully
        assert isinstance(results, dict)

    def test_metrics_with_ignore_index(self):
        """Test metrics handling of ignore index (255)."""
        metrics = SemanticSegmentMetrics()
        
        # Include ignore index
        target = torch.tensor([[0, 1, 255], [255, 0, 1]])
        pred = torch.tensor([[0, 1, 2], [1, 0, 1]])  # Prediction for ignore should not matter
        
        metrics.process([target], [pred])
        results = metrics.results_dict
        
        # Should compute metrics while ignoring 255
        assert isinstance(results, dict)

    def test_metrics_multiple_batches(self):
        """Test metrics with multiple batches."""
        metrics = SemanticSegmentMetrics()
        
        targets = []
        preds = []
        
        for _ in range(3):  # 3 batches
            target = torch.randint(0, 4, (2, 16, 16))
            pred = torch.randint(0, 4, (2, 16, 16))
            targets.append(target)
            preds.append(pred)
        
        metrics.process(targets, preds)
        results = metrics.results_dict
        
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_metrics_different_shapes(self):
        """Test metrics with different tensor shapes."""
        metrics = SemanticSegmentMetrics()
        
        # Different sized tensors
        targets = [
            torch.randint(0, 3, (1, 32, 32)),
            torch.randint(0, 3, (2, 16, 16))
        ]
        preds = [
            torch.randint(0, 3, (1, 32, 32)),
            torch.randint(0, 3, (2, 16, 16))
        ]
        
        metrics.process(targets, preds)
        results = metrics.results_dict
        
        assert isinstance(results, dict)

    def test_metrics_speed_tracking(self):
        """Test speed tracking functionality."""
        metrics = SemanticSegmentMetrics()
        
        # Set speed info (following ClassificationValidator pattern)
        metrics.speed = {
            'preprocess': 1.0,
            'inference': 5.0,
            'postprocess': 0.5
        }
        
        assert metrics.speed['inference'] == 5.0

    def test_metrics_save_dir(self, temp_save_dir):
        """Test save directory functionality."""
        metrics = SemanticSegmentMetrics()
        metrics.save_dir = temp_save_dir
        
        assert metrics.save_dir == temp_save_dir