_base_ = './QDTrack-frcnn_r50_fpn_12e.py'
# model settings
model = dict(
    type='JointMaskQuasiDenseMaskRCNN',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3)
    ),

    roi_head=dict(
        type='JointMaskQuasiDenseRoIHead',
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHeadPlus',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=24,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),  # visible_mask_head loss
        amodal_mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        amodal_mask_head=dict(
            type='JointFCNMaskHeadPlus',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=24,
            loss_amodal_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),  # amodal_mask_head loss
    tracker=dict(type='JointMaskQuasiDenseSegEmbedTracker'),

    # model training and testing settings
    train_cfg=dict(
        rcnn=dict(mask_size=28)),
    test_cfg=dict(
        rcnn=dict(mask_thr_binary=0.5))
)

dataset_type = 'SAILVOSVideoDataset'
data_root = '../data/sailvos_complete_video/'   # SAIL-VOS
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(
        type='SeqLoadAnnotations',
        with_bbox=True,
        with_ins_id=True,
        with_mask=True),
    dict(type='SeqResize', img_scale=(1280, 800), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(
        type='SeqNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='SeqPad', size_divisor=32),
    dict(type='SeqDefaultFormatBundle'),
    dict(
        type='SeqCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices', 'gt_masks'],
        ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,  # 4 for resnext101, 8 for resnet50
    workers_per_gpu=4,  # 4 for resnext101, 8 for resnet50
    train=[
        dict(
            type='SAILVOSVideoDataset',


            # VATrack joint training on SAIL-VOS (amo+vis bbox, amo+vis mask) (contains abrupt scene change)
            ann_file="../data/sailvos_complete_json/joint/train(bbox=amdl_bbox).json",
            img_prefix='../data/sailvos_complete_png/',

            key_img_sampler=dict(interval=1),
            ref_img_sampler=dict(num_ref_imgs=1, scope=3, method='uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations',
                    with_bbox=True,
                    with_ins_id=True,
                    with_mask=True),
                dict(type='SeqResize', img_scale=(1280, 800), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=32),
                dict(type='SeqDefaultFormatBundle'),
                dict(
                    type='SeqCollect',
                    keys=[
                        'img', 'gt_bboxes', 'gt_labels', 'gt_match_indices',
                        'gt_masks'
                    ],
                    ref_prefix='ref')
            ])
    ],
    val=dict(
        type='SAILVOSVideoDataset',


        # VATrack joint validation on SAIL-VOS (amo+vis bbox, amo+vis mask) (contains abrupt scene change)
        ann_file="../data/sailvos_complete_json/joint/valid(bbox=amdl_bbox).json",
        img_prefix='../data/sailvos_complete_png/',

        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='VideoCollect', keys=['img'])
                ])
        ]),
    test=dict(
        type='SAILVOSVideoDataset',


        #VATrack joint testing on SAIL-VOS (amo+vis bbox, amo+vis mask) (contains abrupt scene change)
        ann_file="../data/sailvos_complete_json/joint/valid(bbox=amdl_bbox).json",
        img_prefix='../data/sailvos_complete_png/',

        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='VideoCollect', keys=['img'])
                ])
        ]))
# optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001) # lr=0.01 or 0.005 for bs=8 on resnet50
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)  # lr=0.005 or 0.0025 for bs=4 on resneXt101
## Note: use the smaller lr instead
## in the case of unstable training

optimizer_config = dict(grad_clip=None)
checkpoint_config = dict(interval=1)


evaluation = dict(
    metric=['bbox', 'segm', 'track', 'segtrack'], interval=1)

total_epochs = 12
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=6000,  # 2000, while longer
    warmup_ratio=1.0 / 6000,  # warmup could be used for stable
    step=[8, 11])  # training of QDTrack-mots-joint

workflow = [('train', 1)]
log_level = 'INFO'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

gpu_ids = [0]
load_from = '../ckpts/resnext101_64_4d.pth'
resume_from = None




