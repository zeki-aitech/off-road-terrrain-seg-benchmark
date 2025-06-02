

YAMAHA_SEG_CLASSES_FULL = {
    "RGB_MAP" : {
        (1, 88, 255): 0,    # blue - sky
        (156, 76, 30): 1,   # brown - rough trail
        (178, 176, 153): 2, # grey - smooth trail
        (255, 0, 128): 3,   # pink - slippery trail
        (128, 255, 0): 4,   # bright lime green - traversable grass
        (40, 80, 0): 5,     # dark green - high vegetation
        (0, 160, 0): 6,     # bright green - non-traversable low vegetation
        (255, 0, 0): 7,     # red - obstacle
    },
    "CLASSES" : {
        0: "sky",
        1: "rough_trail",
        2: "smooth_trail",
        3: "slippery_trail",
        4: "traversable_grass",
        5: "high_vegetation",
        6: "non_traversable_low_vegetation",
        7: "obstacle",
    }
}


YAMAHA_SEG_CLASSES_V1 = {
    "RGB_MAP" : {
        (1, 88, 255): 0,    # blue - sky
        (156, 76, 30): 1,   # brown - rough trail
        (178, 176, 153): 2, # grey - smooth trail
        (128, 255, 0): 3,   # bright lime green - traversable grass
        (40, 80, 0): 4,     # dark green - high vegetation
        (0, 160, 0): 5,     # bright green - non-traversable low vegetation
        (255, 0, 0): 6,     # red - obstacle
    },
    "CLASSES" : {
        0: "sky",
        1: "rough_trail",
        2: "smooth_trail",
        3: "traversable_grass",
        4: "high_vegetation",
        5: "non_traversable_low_vegetation",
        6: "obstacle",
    }
}


