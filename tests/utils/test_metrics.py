import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_batch():
    """Create a sample batch following YOLO segmentation format."""
    # Create instance masks with different object IDs
    mask1 = torch.tensor([
        [0, 1, 2, 0],
        [0, 1, 0, 3], 
        [0, 0, 0, 0],
        [4, 0, 0, 0]
    ], dtype=torch.uint8)
    
    mask2 = torch.tensor([
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 2],
        [0, 0, 0, 0]
    ], dtype=torch.uint8)
    
    return {
        "img": torch.randn(2, 3, 64, 64),
        "masks": [mask1, mask2],  # Instance masks with object IDs
        "cls": torch.tensor([5, 7, 12, 3, 8]),  # Class labels for each instance
        "batch_idx": torch.tensor([0, 0, 0, 0, 1])  # Which batch item each instance belongs to
    }


@pytest.fixture
def sample_semantic_masks():
    """Create sample semantic segmentation masks."""
    return torch.tensor([
        [[255, 5, 7, 255],
         [255, 5, 255, 12],
         [255, 255, 255, 255],
         [3, 255, 255, 255]],
        [[255, 8, 255, 255],
         [255, 8, 8, 255],
         [255, 255, 255, 255],
         [255, 255, 255, 255]]
    ], dtype=torch.uint8)


@pytest.fixture
def mock_model():
    """Create a mock model following Ultralytics patterns."""
    class MockSegmentationModel(torch.nn.Module):
        def __init__(self, num_classes=8):
            super().__init__()
            # Follow Ultralytics naming convention for class names
            self.names = {i: f'class_{i}' for i in range(num_classes)}
            self.nc = num_classes
            self.conv = torch.nn.Conv2d(3, num_classes, 1)
            
        def forward(self, x):
            return self.conv(x)
    
    return MockSegmentationModel()


@pytest.fixture
def mock_coco_model():
    """Create a mock model with COCO-style class names."""
    class MockCocoModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.names = {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'
            }
            self.nc = len(self.names)
            self.conv = torch.nn.Conv2d(3, self.nc, 1)
            
        def forward(self, x):
            return self.conv(x)
    
    return MockCocoModel()


@pytest.fixture
def mock_args():
    """Create mock arguments following Ultralytics config patterns."""
    args = Mock()
    args.task = "segment"
    args.half = False
    args.plots = False
    args.workers = 1
    args.batch = 2
    args.split = "val"
    args.data = {
        'names': {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle'},
        'nc': 4
    }
    return args


@pytest.fixture
def sample_predictions():
    """Create sample model predictions (logits)."""
    batch_size, num_classes, height, width = 2, 8, 32, 32
    return torch.randn(batch_size, num_classes, height, width)


@pytest.fixture
def sample_class_predictions():
    """Create sample class predictions (after argmax)."""
    batch_size, height, width = 2, 32, 32
    return torch.randint(0, 8, (batch_size, height, width), dtype=torch.long)


@pytest.fixture
def temp_save_dir(tmp_path):
    """Create a temporary directory for saving test outputs."""
    save_dir = tmp_path / "test_outputs"
    save_dir.mkdir()
    return save_dir


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader for testing."""
    def mock_batch_generator():
        for i in range(2):  # 2 batches
            yield {
                "img": torch.randn(2, 3, 64, 64),
                "masks": [torch.randint(0, 4, (64, 64), dtype=torch.uint8)] * 2,
                "cls": torch.tensor([1, 2, 3]),
                "batch_idx": torch.tensor([0, 0, 1])
            }
    
    mock_dl = Mock()
    mock_dl.__iter__ = mock_batch_generator
    mock_dl.__len__ = Mock(return_value=2)
    mock_dl.dataset = Mock()
    mock_dl.dataset.__len__ = Mock(return_value=4)
    return mock_dl


@pytest.fixture
def mock_metrics_results():
    """Create mock segmentation metrics results."""
    return {
        "mIoU": 0.75,
        "PixelAcc": 0.85,
        "MeanAcc": 0.80
    }


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a session-scoped temporary directory for test data."""
    return tmp_path_factory.mktemp("segmentation_test_data")


@pytest.fixture
def sample_empty_batch():
    """Create a batch with no instances (background only)."""
    empty_mask = torch.zeros((32, 32), dtype=torch.uint8)
    return {
        "img": torch.randn(1, 3, 32, 32),
        "masks": [empty_mask],
        "cls": torch.tensor([]),
        "batch_idx": torch.tensor([])
    }


@pytest.fixture
def large_batch():
    """Create a larger batch for performance testing."""
    masks = []
    cls_labels = []
    batch_indices = []
    
    for batch_idx in range(4):  # 4 items in batch
        mask = torch.randint(0, 5, (128, 128), dtype=torch.uint8)
        masks.append(mask)
        # Add some class labels for this batch item
        num_instances = torch.randint(1, 6, (1,)).item()
        cls_labels.extend([torch.randint(0, 8, (1,)).item() for _ in range(num_instances)])
        batch_indices.extend([batch_idx] * num_instances)
    
    return {
        "img": torch.randn(4, 3, 128, 128),
        "masks": masks,
        "cls": torch.tensor(cls_labels),
        "batch_idx": torch.tensor(batch_indices)
    }


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def mock_logger():
    """Mock the LOGGER to capture log messages in tests."""
    with pytest.MonkeyPatch().context() as m:
        mock_log = Mock()
        m.setattr("src.utils.loss.LOGGER", mock_log)
        m.setattr("src.models.deeplabv3plus.val.LOGGER", mock_log)
        yield mock_log


# Parametrized fixtures for testing different configurations
@pytest.fixture(params=[torch.uint8, torch.int32, torch.long])
def mask_dtype(request):
    """Test different mask data types."""
    return request.param


@pytest.fixture(params=[255, 0, -1])
def ignore_index(request):
    """Test different ignore index values."""
    return request.param


@pytest.fixture(params=[1, 2, 4])
def batch_size(request):
    """Test different batch sizes."""
    return request.param


@pytest.fixture(params=[(32, 32), (64, 64), (128, 128)])
def image_size(request):
    """Test different image sizes."""
    return request.param


# GPU-specific fixtures
@pytest.fixture
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture
def gpu_device():
    """Get GPU device if available, skip test if not."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device('cuda')


# Configuration fixtures following Ultralytics patterns
@pytest.fixture
def ultralytics_config():
    """Create configuration matching Ultralytics format."""
    return {
        'model': 'deeplabv3plus_resnet50.yaml',
        'data': 'coco8-seg.yaml',
        'imgsz': 640,
        'batch': 2,
        'epochs': 1,
        'device': 'cpu',
        'workers': 1,
        'project': 'test_runs',
        'name': 'test_experiment',
        'save': True,
        'plots': False,
        'verbose': False
    }


@pytest.fixture
def mock_validator_callbacks():
    """Create mock callbacks for validator testing."""
    callbacks = {
        'on_val_start': Mock(),
        'on_val_batch_start': Mock(),
        'on_val_batch_end': Mock(),
        'on_val_end': Mock()
    }
    return callbacks


# Error simulation fixtures
@pytest.fixture
def corrupted_batch():
    """Create a batch with intentional issues for error testing."""
    return {
        "img": torch.randn(2, 3, 64, 64),
        "masks": [torch.tensor([[0, 1], [0, 0]])],  # Too small mask
        "cls": torch.tensor([1, 2, 3, 4, 5]),  # Too many classes
        "batch_idx": torch.tensor([0, 0, 1, 1])  # Mismatched indices
    }