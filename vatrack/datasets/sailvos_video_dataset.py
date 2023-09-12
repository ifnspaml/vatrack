from mmdet.datasets import DATASETS

from .coco_video_dataset import CocoVideoDataset


@DATASETS.register_module()
class SAILVOSVideoDataset(CocoVideoDataset):

    CLASSES = (
        'person', 'car', 'motorcycle', 'truck', 'bird', 'dog', 'handbag', 'suitcase', 'bottle', 'cup', 'bowl', 'chair',
        'potted plant', 'bed', 'dining table', 'tv', 'laptop', 'cell phone', 'bag', 'bin', 'box', 'door',
        'road barrier', 'stick')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
