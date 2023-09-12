_base_ = './QDTrack-frcnn_r50_fpn_12e.py'
# model settings
model = dict(
    type='JointBBoxMaskQuasiDenseMaskRCNN',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3)
    ),

    roi_head=dict(
        type='JointBBoxMaskQuasiDenseRoIHead',

        amodal_bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        amodal_bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=24,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, class_weight=[

                    0.1, 0.8, 12.5, 7.0, 15.5, 16.0, 9.5, 20.8, 0.9, 0.9, 3.5, 0.8,  # class weighing
                    1.5, 17.8, 1.9, 3.4, 104.0, 4.4, 1.8, 1.4, 0.35, 0.8, 1.3, 3.3, 1.0  # the last is background

                ],
                loss_weight=1.0),  # amodal_bbox_head cls loss
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),  # amodal_bbox_head bbox loss

        bbox_head=dict(loss_cls=dict(loss_weight=1.0), loss_bbox=dict(loss_weight=1.0)),
        # visible_bbox_head cls/bbox loss

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
    tracker=dict(type='JointBBoxMaskQuasiDenseSegEmbedTracker'),

    # model training and testing settings
    train_cfg=dict(
        rcnn=dict(mask_size=28)),
    test_cfg=dict(
        rcnn=dict(mask_thr_binary=0.5))
)

dataset_type = 'SAILVOSVideoDataset'
data_root = '../data/sailvos_cut_png/'  # SAIL-VOScut
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

            # VATrack with 2 bbox heads joint training on SAIL-VOScut (amo+vis bbox, amo+vis mask)
            ann_file="../data/sailvos_cut_json/joint/train(bbox=both_bbox).json",
            img_prefix='../data/sailvos_cut_png/',


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

        # VATrack with 2 bbox heads joint validation on SAIL-VOScut (amo+vis bbox, amo+vis mask)
        ann_file="../data/sailvos_cut_json/joint/valid(bbox=both_bbox).json",
        img_prefix='../data/sailvos_cut_png/',



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
        # VATrack with 2 bbox heads joint testing on SAIL-VOScut (amo+vis bbox, amo+vis mask)
        ann_file="../data/sailvos_cut_json/joint/valid(bbox=both_bbox).json",
        img_prefix='../data/sailvos_cut_png/',

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
    warmup_iters=6000,  # longer warmup for stable training of
    warmup_ratio=1.0 / 6000,  # QDTrack-mots-joint+
    step=[8, 11])

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




