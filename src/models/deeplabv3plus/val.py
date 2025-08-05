from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn.functional as F

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import plot_images

from src.utils.metrics import SemanticSegmentMetrics
from src.utils.mask_processing import convert_instance_masks_to_semantic


class DeepLabV3PlusSemanticSegmentationValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a DeepLabV3+ segmentation model.

    This validator handles the validation process for semantic segmentation models, including metrics calculation,
    confusion matrix generation, and visualization of results.

    Attributes:
        targets (List[torch.Tensor]): Ground truth segmentation masks.
        pred (List[torch.Tensor]): Model predictions.
        metrics (SemanticSegmentMetrics): Object to calculate and store segmentation metrics.
        names (dict): Mapping of class indices to class names.
        nc (int): Number of classes.

    Methods:
        get_desc: Return a formatted string summarizing segmentation metrics.
        init_metrics: Initialize class names and tracking containers for predictions and targets.
        preprocess: Preprocess input batch by moving data to device and processing masks.
        update_metrics: Update running metrics with model predictions and batch targets.
        finalize_metrics: Finalize metrics including processing speed.
        postprocess: Extract and process the primary prediction from model output.
        get_stats: Calculate and return a dictionary of metrics.
        build_dataset: Create a segmentation dataset instance for validation.
        get_dataloader: Build and return a data loader for segmentation validation.
        print_results: Print evaluation metrics for the segmentation model.
        plot_val_samples: Plot validation image samples with their ground truth masks.
        plot_predictions: Plot images with their predicted segmentation masks.

    Examples:
        >>> from src.models.deeplabv3plus import DeepLabV3PlusSemanticSegmentationValidator
        >>> args = dict(model="deeplabv3plus.pt", data="coco8-seg")
        >>> validator = DeepLabV3PlusSemanticSegmentationValidator(args=args)
        >>> validator()

    Notes:
        This validator is specifically designed for semantic segmentation tasks using DeepLabV3+ architecture.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize DeepLabV3PlusSemanticSegmentationValidator with dataloader, save directory, and other parameters.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (str | Path, optional): Directory to save results.
            args (dict, optional): Arguments containing model and validation configuration.
            _callbacks (list, optional): List of callback functions to be called during validation.

        Examples:
            >>> from src.models.deeplabv3plus import DeepLabV3PlusSemanticSegmentationValidator
            >>> args = dict(model="deeplabv3plus.pt", data="coco8-seg")
            >>> validator = DeepLabV3PlusSemanticSegmentationValidator(args=args)
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.targets = None
        self.pred = None
        self.args.task = "segment"
        self.metrics = SemanticSegmentMetrics()

    def get_desc(self) -> str:
        """Return a formatted string summarizing segmentation metrics."""
        return ("%22s" + "%11s" * 3) % ("classes", "mIoU", "PixelAcc", "MeanAcc")

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize class names and tracking containers for predictions and targets."""
        self.names = model.names
        self.nc = len(model.names)
        self.pred = []
        self.targets = []
        # Initialize segmentation metrics with model info (following ClassificationValidator pattern)
        self.metrics.names = list(model.names.values()) if hasattr(model.names, 'values') else list(model.names)
        self.metrics.nc = self.nc

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess input batch by moving data to device and converting to appropriate dtype."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        
        # Convert instance masks to semantic segmentation format
        batch["semantic_masks"] = convert_instance_masks_to_semantic(batch).to(self.device)
        
        return batch

    def update_metrics(self, preds: torch.Tensor, batch: Dict[str, Any]) -> None:
        """
        Update running metrics with model predictions and batch targets.

        Args:
            preds (torch.Tensor): Model predictions, typically logits with shape [B, C, H, W].
            batch (Dict[str, Any]): Batch data containing images and segmentation masks.

        Notes:
            This method converts logits to class predictions using argmax and resizes targets to match
            prediction dimensions for consistent metric calculation.
        """
        # Get processed semantic masks
        targets = batch["semantic_masks"]  # [B, H, W]
        
        # Convert logits to class predictions if needed
        if preds.dim() == 4 and preds.size(1) > 1:  # [B, C, H, W] with multiple classes
            preds = torch.argmax(preds, dim=1)  # [B, H, W]
        
        # Resize targets to match prediction spatial dimensions
        if targets.shape[-2:] != preds.shape[-2:]:
            targets = F.interpolate(
                targets.float().unsqueeze(1), 
                size=preds.shape[-2:], 
                mode='nearest'
            ).squeeze(1).long()
        
        # Store predictions and targets for metric calculation
        self.pred.append(preds.cpu())
        self.targets.append(targets.cpu())

    def finalize_metrics(self) -> None:
        """
        Finalize metrics including processing speed.

        Notes:
            This method processes the accumulated predictions and targets to generate segmentation metrics,
            and updates the metrics object with speed information.
        """
        # Process segmentation metrics (equivalent to confusion_matrix.process_cls_preds)
        if self.targets and self.pred:
            self.metrics.process(self.targets, self.pred)
        
        # Add speed and save directory info (following ClassificationValidator pattern)
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir

    def postprocess(self, preds: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]) -> torch.Tensor:
        """Extract and process the primary prediction from model output."""
        # Extract first tensor if predictions are in list/tuple format
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        
        # Convert logits to class predictions if needed
        if preds.dim() == 4 and preds.size(1) > 1:  # [B, C, H, W]
            preds = torch.argmax(preds, dim=1)  # [B, H, W]
        
        return preds

    def get_stats(self) -> Dict[str, float]:
        """Calculate and return a dictionary of metrics by processing targets and predictions."""
        return self.metrics.results_dict

    def build_dataset(self, img_path: str):
        """Create a segmentation dataset instance for validation."""
        return build_yolo_dataset(
            cfg=self.args,
            img_path=img_path,
            batch=self.args.batch,
            data=getattr(self.args, 'data', {}),
            mode='val',
        )

    def get_dataloader(self, dataset_path: Union[Path, str], batch_size: int) -> torch.utils.data.DataLoader:
        """
        Build and return a data loader for segmentation validation.

        Args:
            dataset_path (str | Path): Path to the dataset directory.
            batch_size (int): Number of samples per batch.

        Returns:
            (torch.utils.data.DataLoader): DataLoader object for the segmentation validation dataset.
        """
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(
            dataset=dataset,
            batch=batch_size,
            workers=getattr(self.args, 'workers', 8),
            shuffle=False,
            rank=-1
        )

    def print_results(self) -> None:
        """Print evaluation metrics for the segmentation model."""
        stats = self.get_stats()
        if stats:
            pf = "%22s" + "%11.3g" * len(stats)
            LOGGER.info(pf % ("all", *stats.values()))

    def plot_val_samples(self, batch: Dict[str, Any], ni: int) -> None:
        """
        Plot validation image samples with their ground truth segmentation masks.

        Args:
            batch (Dict[str, Any]): Dictionary containing batch data with 'img' and 'semantic_masks'.
            ni (int): Batch index used for naming the output file.

        Examples:
            >>> validator = DeepLabV3PlusSemanticSegmentationValidator()
            >>> batch = {"img": torch.rand(4, 3, 640, 640), "semantic_masks": torch.randint(0, 20, (4, 640, 640))}
            >>> validator.plot_val_samples(batch, 0)
        """
        # Skip plotting for segmentation to avoid format conflicts with plot_images
        if not self.args.plots:
            return
        LOGGER.info(f"Segmentation validation sample plotting skipped for batch {ni}")

    def plot_predictions(self, batch: Dict[str, Any], preds: torch.Tensor, ni: int) -> None:
        """
        Plot images with their predicted segmentation masks and save the visualization.

        Args:
            batch (Dict[str, Any]): Batch data containing images and other information.
            preds (torch.Tensor): Model predictions with shape (batch_size, height, width).
            ni (int): Batch index used for naming the output file.

        Examples:
            >>> validator = DeepLabV3PlusSemanticSegmentationValidator()
            >>> batch = {"img": torch.rand(4, 3, 640, 640)}
            >>> preds = torch.randint(0, 20, (4, 640, 640))
            >>> validator.plot_predictions(batch, preds, 0)
        """
        # Skip plotting for segmentation to avoid format conflicts with plot_images
        if not self.args.plots:
            return
        LOGGER.info(f"Segmentation prediction plotting skipped for batch {ni}")