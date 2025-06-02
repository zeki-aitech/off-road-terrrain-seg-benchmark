## Production Target Summary for AMR Off-Road Segmentation

The primary goals are **safety, reliability, and sufficient accuracy for navigation.**

1.  **Overall Segmentation Accuracy (Mask mAP50 / mIoU at 0.5 IoU):**
    *   **Target: 0.75 - 0.85+**
    *   *Reasoning: Significant improvement needed for reliable overall scene understanding.*

2.  **Critical Class Performance (Mask mAP50 / mIoU at 0.5 IoU):**
    *   **Traversable Areas** (e.g., `smooth_trail`, `traversable_grass`):
        *   **Target: 0.85 - 0.95+**
        *   *Reasoning: Highest priority for safe path planning. Must minimize both missing safe paths (false negatives) and incorrectly identifying unsafe paths as safe (false positives).*
    *   **Obstacles & Non-Traversable Areas** (e.g., `obstacle`, `non_traversable_low_vegetation`, `rough_trail`):
        *   **Target: 0.80 - 0.90+** (especially critical for `obstacle`)
        *   *Reasoning: Essential for collision avoidance and preventing the AMR from getting stuck.*

3.  **Localization Precision (Mask mAP50-95 / mIoU averaged over 0.5-0.95 IoU):**
    *   **Target: 0.50 - 0.60+** (higher for critical classes)
    *   *Reasoning: Indicates how accurately the model outlines objects. Better precision means more reliable interaction with object boundaries.*

4.  **Specific Safety Metrics (Track Explicitly):**
    *   **False Positive Rate for Traversable Areas:** Target **Extremely Low** (i.e., very rarely label an unsafe area as traversable).
    *   **False Negative Rate for Obstacles:** Target **Extremely Low** (i.e., very rarely miss detecting an actual obstacle).

5.  **Inference Speed:**
    *   **Target: Maintain** real-time performance (e.g., inference < ~20ms, total pipeline < 30-50ms, or as per AMR's operational requirements).
    *   *Reasoning: Must be fast enough for the AMR to react to its environment.*

6.  **Robustness & Generalization:**
    *   **Target: High and consistent performance** across diverse off-road conditions (lighting, weather, terrain types, seasons). This is tested through rigorous and varied validation datasets.

**In short, aim for significant improvements in accuracy and reliability for critical classes like traversable paths and obstacles, and better overall localization precision, while maintaining excellent inference speed.**

---

Alternatively, here's a table summarizing the key quantitative targets:

| Metric Category                      | Target Value (Mask mAP50 or mAP50-95) | Notes                                                                 |
|--------------------------------------|---------------------------------------|-----------------------------------------------------------------------|
| **Overall Segmentation Accuracy**    | **0.75 - 0.85+**                      | For all classes combined                                              |
| **Traversable Areas**                | **0.85 - 0.95+**                      | e.g., `smooth_trail`, `traversable_grass`                           |
| **Obstacles & Non-Traversable**      | **0.80 - 0.90+**                      | Especially critical for the `obstacle` class                            |
| **Localization Precision (Overall)** | **0.50 - 0.60+**                      | How precisely objects are outlined                                    |
| **Inference Speed (Total/Image)**    | **Maintain current or <30-50ms**      | Ensure it meets AMR's real-time needs                                 |
| **False Positives (Traversable)**    | **Extremely Low**                     | Labeling unsafe area as safe                                          |
| **False Negatives (Obstacles)**      | **Extremely Low**                     | Missing an actual obstacle                                            |

