"""
Monkey patches for Ultralytics to support custom models.
"""

def apply_patches():
    """Apply monkey patches to replace Ultralytics functions with custom implementations."""
    import ultralytics.nn.tasks
    from src.nn.tasks import parse_model
    
    # Replace Ultralytics parse_model with our custom implementation
    ultralytics.nn.tasks.parse_model = parse_model
    
    print("✅ Monkey patches applied: parse_model replaced with DeepLabV3+ version")