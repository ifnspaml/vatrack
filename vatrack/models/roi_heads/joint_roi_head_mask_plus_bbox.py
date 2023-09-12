import torch
from mmdet.core import bbox2roi, build_assigner, build_sampler
from mmdet.models import HEADS, build_head, build_roi_extractor
from mmdet.models.roi_heads import StandardRoIHead
from .base_joint_roi_head import BaseJointRoIHead
from .amodal_test_mixins import AmodalMaskTestMixin
from .amodal_test_mixins import AmodalBBoxTestMixin
import random


@HEADS.register_module()
class JointBBoxMaskQuasiDenseRoIHead(BaseJointRoIHead, AmodalMaskTestMixin, AmodalBBoxTestMixin, StandardRoIHead):
    """
    That is VATrack 2bbox RoI head.
    """
    def __init__(self,
                 track_roi_extractor=None,
                 track_head=None,
                 track_train_cfg=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert track_head is not None
        self.track_train_cfg = track_train_cfg
        self.init_track_head(track_roi_extractor, track_head)
        if self.track_train_cfg:
            self.init_track_assigner_sampler()
        assert self.mask_head is not None
        assert self.amodal_mask_head is not None

        self.test_cfg_v = self.test_cfg.deepcopy()
        self.test_cfg_v['nms']['iou_threshold'] = 0.999

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_amodal_bbox_head(self, amodal_bbox_roi_extractor, amodal_bbox_head):
        """Initialize ``bbox_head``"""
        self.amodal_bbox_roi_extractor = build_roi_extractor(amodal_bbox_roi_extractor)
        self.amodal_bbox_head = build_head(amodal_bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_track_assigner_sampler(self):
        """Initialize assigner and sampler."""
        if self.track_train_cfg.get('assigner', None):
            self.track_roi_assigner = build_assigner(
                self.track_train_cfg.assigner)
            self.track_share_assigner = False
        else:
            self.track_roi_assigner = self.bbox_assigner
            self.track_share_assigner = True

        if self.track_train_cfg.get('sampler', None):
            self.track_roi_sampler = build_sampler(
                self.track_train_cfg.sampler, context=self)
            self.track_share_sampler = False
        else:
            self.track_roi_sampler = self.bbox_sampler
            self.track_share_sampler = True

    def init_amodal_mask_head(self, amodal_mask_roi_extractor, amodal_mask_head):
        """Initialize ``amodal_mask_head``"""
        if amodal_mask_roi_extractor is not None:
            self.amodal_mask_roi_extractor = build_roi_extractor(amodal_mask_roi_extractor)
            self.amodal_share_roi_extractor = False
        else:
            self.amodal_share_roi_extractor = True
            self.amodal_mask_roi_extractor = self.bbox_roi_extractor
        self.amodal_mask_head = build_head(amodal_mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            amodal_bbox_results = self._amodal_bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred']) + (amodal_bbox_results['cls_score'],
                                                         amodal_bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results_v = self._mask_forward(x, mask_rois)
            mask_results_a = self._amodal_forward(x, mask_rois)
            outs = outs + ([mask_results_v['mask_pred'], mask_results_a['mask_pred']], )
        return outs

    @property
    def with_track(self):
        """bool: whether the RoI head contains a `track_head`"""
        return hasattr(self, 'track_head') and self.track_head is not None

    def init_track_head(self, track_roi_extractor, track_head):
        """Initialize ``track_head``"""
        if track_roi_extractor is not None:
            self.track_roi_extractor = build_roi_extractor(track_roi_extractor)
            self.track_share_extractor = False
        else:
            self.track_share_extractor = True
            self.track_roi_extractor = self.bbox_roi_extractor
        self.track_head = build_head(track_head)

    def init_weights(self, *args, **kwargs):
        super().init_weights(*args, **kwargs)
        if self.with_track:
            self.track_head.init_weights()
            if not self.track_share_extractor:
                self.track_roi_extractor.init_weights()

        if self.with_bbox:
            self.amodal_bbox_roi_extractor.init_weights()
            self.amodal_bbox_head.init_weights()
        if self.with_mask:
            self.amodal_mask_head.init_weights()
            if not self.share_roi_extractor:
                self.amodal_mask_roi_extractor.init_weights()

    def _amodal_bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.amodal_bbox_roi_extractor(
            x[:self.amodal_bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.amodal_bbox_head(bbox_feats)

        amodal_bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return amodal_bbox_results

    def _amodal_bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        amodal_bbox_results = self._amodal_bbox_forward(x, rois)

        bbox_targets = self.amodal_bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_amodal_bbox = self.amodal_bbox_head.loss(amodal_bbox_results['cls_score'],
                                        amodal_bbox_results['bbox_pred'], rois,
                                        *bbox_targets)
        loss_amodal_bbox["loss_amodal_bbox"] = loss_amodal_bbox['loss_bbox']
        loss_amodal_bbox["loss_amodal_cls"] = loss_amodal_bbox['loss_cls']
        loss_amodal_bbox["acc_amodal"] = loss_amodal_bbox['acc']
        del loss_amodal_bbox['loss_bbox'], loss_amodal_bbox['loss_cls'], loss_amodal_bbox['acc']

        amodal_bbox_results.update(loss_amodal_bbox=loss_amodal_bbox)
        return amodal_bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks_v,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:           # no share_roi_extractor
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            if pos_rois.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            if pos_inds.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks_v,
                                                  self.train_cfg)

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(                # mask_roi_extractor with output_size=14
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:                            # false
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)

        return mask_results

    def _amodal_forward_train(self, x, sampling_results, bbox_feats, gt_masks_a,
                            img_metas):
        """Run forward function and calculate loss for amodal mask head in
        training."""
        if not self.amodal_share_roi_extractor:           # no share_roi_extractor
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            if pos_rois.shape[0] == 0:
                return dict(loss_amodal_mask=None)
            amodal_mask_results = self._amodal_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            if pos_inds.shape[0] == 0:
                return dict(loss_amodal_mask=None)
            amodal_mask_results = self._amodal_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        amodal_mask_targets = self.amodal_mask_head.get_targets(sampling_results, gt_masks_a,
                                                  self.train_cfg)

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_amodal_mask = self.amodal_mask_head.loss(amodal_mask_results['mask_pred'],
                                        amodal_mask_targets, pos_labels)

        amodal_mask_results.update(loss_amodal_mask=loss_amodal_mask, amodal_mask_targets=amodal_mask_targets)
        return amodal_mask_results

    def _amodal_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Amodal Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            amodal_mask_feats = self.amodal_mask_roi_extractor(          # mask_roi_extractor with output_size=14
                x[:self.amodal_mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:                                    # false
                amodal_mask_feats = self.shared_head(amodal_mask_feats)
        else:
            assert bbox_feats is not None
            amodal_mask_feats = bbox_feats[pos_inds]

        amodal_mask_pred = self.amodal_mask_head(amodal_mask_feats)
        amodal_mask_results = dict(mask_pred=amodal_mask_pred, mask_feats=amodal_mask_feats)

        return amodal_mask_results


    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list, # key amodal proposals
                      gt_bboxes, ##
                      gt_labels,
                      gt_match_indices,
                      ref_x,
                      ref_img_metas,
                      ref_proposals, # reference amodal proposals
                      ref_gt_bboxes, ##
                      ref_gt_labels,
                      gt_bboxes_ignore=None, ##
                      gt_masks=None, ##
                      ref_gt_bboxes_ignore=None, ##
                      *args,
                      **kwargs):
        #losses = super().forward_train(x, img_metas, proposal_list, gt_bboxes,
        #                               gt_labels, gt_bboxes_ignore, gt_masks)

        if isinstance(gt_masks, list):
            gt_masks_a = [gt_masks[i][0] for i in range(len(gt_masks))]
            gt_masks_v = [gt_masks[i][1] for i in range(len(gt_masks))]
        else:
            print('gt_masks is not list')
            exit()

        if isinstance(gt_bboxes, list):
            gt_bboxes_a = [gt_bboxes[i][0] for i in range(len(gt_bboxes))]
            gt_bboxes_v = [gt_bboxes[i][1] for i in range(len(gt_bboxes))]
        else:
            print('gt_bboxes is not list')
            exit()

        if isinstance(gt_bboxes_ignore, list):
            gt_bboxes_ignore_a = [gt_bboxes_ignore[i][0] for i in range(len(gt_bboxes_ignore))]
            gt_bboxes_ignore_v = [gt_bboxes_ignore[i][1] for i in range(len(gt_bboxes_ignore))]
            print('gt_bboxes_ignore is not none')
        else:
            gt_bboxes_ignore_a = None
            gt_bboxes_ignore_v = None

        if isinstance(ref_gt_bboxes, list):
            ref_gt_bboxes_a = [ref_gt_bboxes[i][0] for i in range(len(ref_gt_bboxes))]
            ref_gt_bboxes_v = [ref_gt_bboxes[i][1] for i in range(len(ref_gt_bboxes))]
        else:
            print('ref_gt_bboxes is not list')
            exit()

        if isinstance(ref_gt_bboxes_ignore, list):
            ref_gt_bboxes_ignore_a = [ref_gt_bboxes_ignore[i][0] for i in range(len(ref_gt_bboxes_ignore))]
            ref_gt_bboxes_ignore_v = [ref_gt_bboxes_ignore[i][1] for i in range(len(ref_gt_bboxes_ignore))]
            print('ref_gt_bboxes_ignore is not none')
        else:
            ref_gt_bboxes_ignore_a = None
            ref_gt_bboxes_ignore_v = None


        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore_a is None:
                gt_bboxes_ignore_a = [None for _ in range(num_imgs)]
            if gt_bboxes_ignore_v is None:
                gt_bboxes_ignore_v = [None for _ in range(num_imgs)]

            amodal_sampling_results, visible_sampling_results, sampling_results_bbox = [], [], []
            for i in range(num_imgs):
                amodal_assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes_a[i], gt_bboxes_ignore_a[i],
                    gt_labels[i])
                amodal_sampling_result = self.bbox_sampler.sample(
                    amodal_assign_result,
                    proposal_list[i],
                    gt_bboxes_a[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                amodal_sampling_results.append(amodal_sampling_result)

            for i in range(num_imgs):
                visible_assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes_v[i], gt_bboxes_ignore_v[i],
                    gt_labels[i])
                visible_sampling_result = self.bbox_sampler.sample(
                    visible_assign_result,
                    proposal_list[i],
                    gt_bboxes_v[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                visible_sampling_results.append(visible_sampling_result)

            # proposal_list_a = torch.zeros(0, 5).to(device="cuda:0")
            #
            # rdm_idx = []
            # for i in range(num_imgs):
            #     if gt_bboxes_a[i].shape[0] > 0:
            #         rdm_idx_ = random.randint(0, gt_bboxes_a[i].shape[0]-1)
            #         rdm_idx.append(rdm_idx_)
            #         assign_result_bbox = self.bbox_assigner.assign(
            #             proposal_list_a, gt_bboxes_a[i].split(1, 0)[rdm_idx_], gt_bboxes_ignore_v[i],
            #             gt_labels[i].split(1, 0)[rdm_idx_])
            #         sampling_result_bbox = self.bbox_sampler.sample(
            #             assign_result_bbox,
            #             proposal_list_a,
            #             gt_bboxes_a[i].split(1, 0)[rdm_idx_],
            #             gt_labels[i].split(1, 0)[rdm_idx_],
            #             feats=[lvl_feat[i][None] for lvl_feat in x])
            #         sampling_results_bbox.append(sampling_result_bbox)
            #     else:
            #         rdm_idx.append(None)
            #         assign_result_bbox = self.bbox_assigner.assign(
            #             proposal_list_a, gt_bboxes_a[i], gt_bboxes_ignore_v[i],
            #             gt_labels[i])
            #         sampling_result_bbox = self.bbox_sampler.sample(
            #             assign_result_bbox,
            #             proposal_list_a,
            #             gt_bboxes_a[i],
            #             gt_labels[i],
            #             feats=[lvl_feat[i][None] for lvl_feat in x])
            #         sampling_results_bbox.append(sampling_result_bbox)
            #
            # for i in range(num_imgs):
            #     assign_result_bbox = self.bbox_assigner.assign(
            #         proposal_list_a, gt_bboxes_a[i], gt_bboxes_ignore_a[i],
            #         gt_labels[i])
            #     sampling_result_bbox = self.bbox_sampler.sample(
            #         assign_result_bbox,
            #         proposal_list_a,
            #         gt_bboxes_a[i],
            #         gt_labels[i],
            #         feats=[lvl_feat[i][None] for lvl_feat in x])
            #     sampling_results_bbox.append(sampling_result_bbox)
            #
            # for i in range(num_imgs):
            #     visible_assign_result = self.bbox_assigner.assign(
            #         proposal_list_a, gt_bboxes_v[i], gt_bboxes_ignore_v[i],
            #         gt_labels[i])
            #     visible_sampling_result = self.bbox_sampler.sample(
            #         visible_assign_result,
            #         proposal_list_a,
            #         gt_bboxes_v[i],
            #         gt_labels[i],
            #         feats=[lvl_feat[i][None] for lvl_feat in x])
            #     visible_sampling_results.append(visible_sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            amodal_bbox_results = self._amodal_bbox_forward_train(x, amodal_sampling_results,
                                                                  gt_bboxes_a, gt_labels,
                                                                  img_metas)
            losses.update(amodal_bbox_results['loss_amodal_bbox'])

            # gt_bboxes_v_vis_bbox = []
            # for i in range(num_imgs):
            #     if rdm_idx[i] is not None:
            #         gt_bboxes_v_vis_bbox.append(gt_bboxes_v[i].split(1, 0)[rdm_idx[i]])
            #     else:
            #         gt_bboxes_v_vis_bbox.append(torch.zeros(0, 4).to(device="cuda:0"))
            # bbox_results = self._bbox_forward_train(x, sampling_results_bbox,
            #                                         gt_bboxes_v_vis_bbox, gt_labels,
            #                                         img_metas)

            bbox_results = self._bbox_forward_train(x, visible_sampling_results,
                                                    gt_bboxes_v, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])


        # mask head forward and loss
        if self.with_mask:
            mask_results_v = self._mask_forward_train(x, visible_sampling_results,
                                                      bbox_results['bbox_feats'],
                                                      gt_masks_v, img_metas)
            mask_results_a = self._amodal_forward_train(x, amodal_sampling_results,
                                                      amodal_bbox_results['bbox_feats'],
                                                      gt_masks_a, img_metas)
            # TODO: Support empty tensor input. #2280
            if mask_results_v['loss_mask'] is not None:
                losses.update(mask_results_v['loss_mask'])
            if mask_results_a['loss_amodal_mask'] is not None:
                losses.update(mask_results_a['loss_amodal_mask'])
            # if mask_results['loss_dice'] is not None:
            #     losses.update(mask_results['loss_dice'])
            # if mask_results['loss_bound'] is not None:
            #     losses.update(mask_results['loss_bound'])


        # TODO: use amodal proposals to train track head
        num_imgs = len(img_metas)
        if gt_bboxes_ignore_a is None:
            gt_bboxes_ignore_a = [None for _ in range(num_imgs)]
        if gt_bboxes_ignore_v is None:
            gt_bboxes_ignore_v = [None for _ in range(num_imgs)]
        if ref_gt_bboxes_ignore_a is None:
            ref_gt_bboxes_ignore_a = [None for _ in range(num_imgs)]
        if ref_gt_bboxes_ignore_v is None:
            ref_gt_bboxes_ignore_v = [None for _ in range(num_imgs)]
        key_sampling_results, ref_sampling_results = [], []
        for i in range(num_imgs):
            assign_result = self.track_roi_assigner.assign(
                proposal_list[i], gt_bboxes_a[i], gt_bboxes_ignore_a[i],
                gt_labels[i])
            sampling_result = self.track_roi_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes_a[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            key_sampling_results.append(sampling_result)

            ref_assign_result = self.track_roi_assigner.assign(
                ref_proposals[i], ref_gt_bboxes_a[i], ref_gt_bboxes_ignore_a[i],
                ref_gt_labels[i])
            ref_sampling_result = self.track_roi_sampler.sample(
                ref_assign_result,
                ref_proposals[i],
                ref_gt_bboxes_a[i],
                ref_gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in ref_x])
            ref_sampling_results.append(ref_sampling_result)

        ## loss track
        key_bboxes = [res.pos_bboxes for res in key_sampling_results]
        key_feats = self._track_forward(x, key_bboxes)
        ref_bboxes = [res.bboxes for res in ref_sampling_results]
        ref_feats = self._track_forward(ref_x, ref_bboxes)

        # because only using amodal bboxes to train track_head, so the track_head will
        # learn the amodal tracking ability of matching amodal candidate bboxes

        match_feats = self.track_head.match(key_feats, ref_feats,
                                            key_sampling_results,
                                            ref_sampling_results)
        asso_targets = self.track_head.get_track_targets(
            gt_match_indices, key_sampling_results, ref_sampling_results)
        loss_track = self.track_head.loss(*match_feats, *asso_targets)

        losses.update(loss_track)

        return losses

    def _track_forward(self, x, bboxes):
        """Track head forward function used in both training and testing."""
        rois = bbox2roi(bboxes)
        track_feats = self.track_roi_extractor(
            x[:self.track_roi_extractor.num_inputs], rois)
        track_feats = self.track_head(track_feats)
        return track_feats

    def simple_test(self, x, img_metas, proposal_list, rescale):            # complex logic here for ensuring one
        det_bboxes_a, det_labels_a = self.simple_test_amodal_bboxes(        # amo bbox matching to one vis bbox
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        det_bboxes_v_mid = torch.zeros(1, 5).to(device="cuda:0")
        det_bboxes_a_mid = torch.zeros(1, 5).to(device="cuda:0")
        det_labels = torch.zeros(1)
        det_labels = det_labels.int()
        if isinstance(det_bboxes_a[0], torch.Tensor):
            for i in range(det_bboxes_a[0].shape[0]):
                proposal_list_for_v = det_bboxes_a[0][i, :].unsqueeze(0)
                det_labels_for_v = det_labels_a[0][i].unsqueeze(0)
                # proposal_list_v = [proposal_list_v.repeat(1000, 1)]   # no need to repeat 1000 times for proposals
                proposal_list_for_v = [proposal_list_for_v]
                det_bboxes_v_single, det_labels_v_single = self.simple_test_bboxes(
                    x, img_metas, proposal_list_for_v, self.test_cfg_v, rescale=rescale)
                if det_bboxes_v_single[0].shape[0] >= 1:
                    a = det_bboxes_v_single[0].split(1, 0)[0]
                    a[:, 4] = proposal_list_for_v[0].split(1, 0)[0][:, 4]
                    det_bboxes_v_mid = torch.cat((a, det_bboxes_v_mid), 0)
                    det_bboxes_a_mid = torch.cat((proposal_list_for_v[0].split(1, 0)[0], det_bboxes_a_mid), 0)
                    det_labels = torch.cat((det_labels_for_v.split(1, 0)[0], det_labels), 0)
                else:
                    det_bboxes_v_mid = torch.cat((proposal_list_for_v[0].split(1, 0)[0], det_bboxes_v_mid), 0)
                    det_bboxes_a_mid = torch.cat((proposal_list_for_v[0].split(1, 0)[0], det_bboxes_a_mid), 0)
                    det_labels = torch.cat((det_labels_for_v.split(1, 0)[0], det_labels), 0)
            det_bboxes_v = [det_bboxes_v_mid[:-1, :]]
            det_bboxes_a = [det_bboxes_a_mid[:-1, :]]
            det_labels = [det_labels[:-1]]
        else:
            det_bboxes_a = [torch.tensor([[]]).to(device="cuda:0")]
            det_bboxes_v = [torch.tensor([[]]).to(device="cuda:0")]
            det_labels = [torch.tensor([]).to(device="cuda:0")]

        det_bboxes_a = det_bboxes_a[0]
        det_bboxes_v = det_bboxes_v[0]

        det_masks_v = self.simple_test_mask(
            x, img_metas, det_bboxes_v, det_labels, rescale=rescale)
        det_masks_a = self.simple_test_amodal_mask(
            x, img_metas, det_bboxes_a, det_labels, rescale=rescale)

        if det_bboxes_a.size(0) == 0:
            return [det_bboxes_a, det_bboxes_v], det_labels, [det_masks_a, det_masks_v], None

        track_bboxes = det_bboxes_a[:, :-1] * torch.tensor(
            img_metas[0]['scale_factor']).to(det_bboxes_a.device)
        track_feats = self._track_forward(x, [track_bboxes])
        # track feat of detected amodal bboxes, because track_head is trained with amodal GT bboxes in training

        return [det_bboxes_a, det_bboxes_v], det_labels, [det_masks_a, det_masks_v], track_feats

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Simple test for mask head without augmentation."""
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            mask_results = dict(mask_pred=None, mask_feats=None)
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_results = self._mask_forward(x, mask_rois)
        return mask_results

    def simple_test_amodal_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Simple test for mask head without augmentation."""
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            amodal_mask_results = dict(mask_pred=None, mask_feats=None)
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            amodal_mask_results = self._amodal_forward(x, mask_rois)
        return amodal_mask_results

    def get_seg_masks(self, img_metas, det_bboxes, det_labels, det_masks,
                      rescale=False):
        """Simple test for mask head without augmentation."""
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
            det_segms = []
            labels = []
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            segm_result, det_segms, labels = self.mask_head.get_seg_masks(
                det_masks['mask_pred'], _bboxes, det_labels, self.test_cfg,
                ori_shape, scale_factor, rescale)
        return segm_result, det_segms, labels

    def get_seg_amodal_masks(self, img_metas, det_bboxes, det_labels, det_masks,
                      rescale=False):
        """Simple test for mask head without augmentation."""
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.amodal_mask_head.num_classes)]
            det_segms = []
            labels = []
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            segm_result, det_segms, labels = self.amodal_mask_head.get_seg_amodal_masks(
                det_masks['mask_pred'], _bboxes, det_labels, self.test_cfg,
                ori_shape, scale_factor, rescale)
        return segm_result, det_segms, labels
