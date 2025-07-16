import os
from typing import List

from ultralytics.data import BaseDataset

# def img2mask_paths(img_paths: List[str]) -> List[str]:
#     """
#     Convert image paths to mask paths by replacing 'images' with 'masks'
#     and changing the file extension to '.png'.
#     """
#     sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}masks{os.sep}"  # /images/, /masks/ substrings
#     return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".png" for x in img_paths]
    

# class SemanticDataset(BaseDataset):
#     """Semantic Segmentation Dataset."""

#     def __init__(self, *args, task: str = "segment", channels: int = 3, **kwargs):
#         assert task == "segment", "SemanticDataset only supports 'segment' task."
#         super().__init__(*args, channels=channels, **kwargs)
        
#     def get_labels(self):
#         """
#         Return dictionary of labels for Semantic Segmentation.
#         """



     

# if __name__ == "__main__":
#     img_paths = [
#         "/dataset/images/train/img1.jpg",
#         "/dataset/images/val/img2.jpeg",
#         "/dataset/images/train/subfolder/img3.bmp"
#     ]
#     mask_paths = img2mask_paths(img_paths)
#     for img, mask in zip(img_paths, mask_paths):
#         print(f"Image: {img}  -->  Mask: {mask}")
    