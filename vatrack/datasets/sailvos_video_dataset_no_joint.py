from mmdet.datasets import DATASETS

from .coco_video_dataset_no_joint import CocoVideoDatasetNoJoint


@DATASETS.register_module()
class SAILVOSVideoDatasetNoJoint(CocoVideoDatasetNoJoint):

    CLASSES = (
        'person', 'car', 'motorcycle', 'truck', 'bird', 'dog', 'handbag', 'suitcase', 'bottle', 'cup', 'bowl', 'chair',
        'potted plant', 'bed', 'dining table', 'tv', 'laptop', 'cell phone', 'bag', 'bin', 'box', 'door',
        'road barrier', 'stick')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
