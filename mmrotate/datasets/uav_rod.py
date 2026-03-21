# Copyright (c) OpenMMLab. All rights reserved.
from .dota import DOTADataset
from .builder import ROTATED_DATASETS

@ROTATED_DATASETS.register_module()
class UAV_RODDataset(DOTADataset):
    """low-altitude drone-based dataset of UAV-ROD (Support UAV-ROD dataset)."""
    CLASSES = ('car', )
    PALETTE = [
        (255, 0, 0),
    ]
