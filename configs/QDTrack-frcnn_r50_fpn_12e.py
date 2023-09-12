# model settings
model = dict(
    type='EMQuasiDenseFasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),

        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'

        # type='ConvNeXt',              # ConvNeXt could be further implemented
        # arch='tiny',
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint=
        #     'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_3rdparty_32xb128-noema_in1k_20220222-2908964a.pth',
        #     prefix='backbone.'),
        # # drop_path_rate=0.5,
        # # layer_scale_init_value=1.0,
        # # gap_before_final_norm=False,
        # out_indices=(0, 1, 2, 3)
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='QuasiDenseRoIHead',

        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
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

                    0.1, 0.8, 12.5, 7.0, 15.5, 16.0, 9.5, 20.8, 0.9, 0.9, 3.5, 0.8,      # class weighing
                    1.5, 17.8, 1.9, 3.4, 104.0, 4.4, 1.8, 1.4, 0.35, 0.8, 1.3, 3.3, 1.0  # the last is background

                ],                                        loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        track_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        track_head=dict(
            type='QuasiDenseEmbedHead',
            num_convs=4,
            num_fcs=1,
            embed_channels=256,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_track=dict(type='MultiPosCrossEntropyLoss', loss_weight=0.25),
            loss_track_aux=dict(
                type='L2Loss',
                neg_pos_ub=3,
                pos_margin=0,
                neg_margin=0.3,
                hard_mining=True,
                loss_weight=1.0))),
        tracker=dict(
            type='QuasiDenseEmbedTracker',
            # init_score_thr=0.7,
            init_score_thr=0.01,                 # 0.01 for greater recall (better AP results)
            # obj_score_thr=0.15,
            obj_score_thr=0.01,                  # 0.01 for greater recall (better AP results)
            match_score_thr=0.5,
            memo_tracklet_frames=32,
            memo_backdrop_frames=1,
            memo_momentum=0.8,
            nms_conf_thr=0.5,
            nms_backdrop_iou_thr=0.3,
            nms_class_iou_thr=0.7,
            with_cats=True,
            match_metric='bisoftmax'),

        # model training and testing settings
        train_cfg = dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_across_levels=False,
                nms_pre=2000,
                nms_post=1000,
                max_num=1000,
                nms_thr=0.7,
                min_bbox_size=1),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            embed=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='CombinedSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=3,
                    add_gt_as_proposals=True,
                    pos_sampler=dict(type='InstanceBalancedPosSampler'),
                    neg_sampler=dict(
                        type='IoUBalancedNegSampler',
                        floor_thr=-1,
                        floor_fraction=0,
                        num_bins=3)))),
        test_cfg=dict(
            rpn=dict(
                nms_across_levels=False,
                nms_pre=1000,
                nms_post=1000,
                max_num=1000,
                nms_thr=0.7,
                min_bbox_size=1),           # in joint test, there could be some bbox with h=0 or w=0
            rcnn=dict(
                # score_thr=0.15,
                score_thr=0.01,             # here 0.01 to satisfy init_score_thr/obj_score_thr=0.01 in tracker
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100))
)




