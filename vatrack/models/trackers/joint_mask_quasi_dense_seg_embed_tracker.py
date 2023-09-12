import torch
import torch.nn.functional as F
from mmdet.core import bbox_overlaps

from ..builder import TRACKERS
from .quasi_dense_embed_tracker import QuasiDenseEmbedTracker


@TRACKERS.register_module()
class JointMaskQuasiDenseSegEmbedTracker(QuasiDenseEmbedTracker):
    """
    That is VATrack Tracker.
    """
    def update_memo(self, ids, bboxes, mask_preds_a, mask_preds_v, embeds, labels, frame_id):
        tracklet_inds = ids > -1
        # update memo
        for id, bbox, mask_pred_a, mask_pred_v, embed, label in (
            zip(
                ids[tracklet_inds],
                bboxes[tracklet_inds],
                mask_preds_a[tracklet_inds],
                mask_preds_v[tracklet_inds],
                embeds[tracklet_inds],
                labels[tracklet_inds])):
            id = int(id)
            if id in self.tracklets.keys():
                velocity = (bbox - self.tracklets[id]['bbox']) / (
                    frame_id - self.tracklets[id]['last_frame'])
                self.tracklets[id]['bbox'] = bbox
                self.tracklets[id]['mask_pred_a'] = mask_pred_a
                self.tracklets[id]['mask_pred_v'] = mask_pred_v
                self.tracklets[id]['embed'] = (
                    1 - self.memo_momentum
                ) * self.tracklets[id]['embed'] + self.memo_momentum * embed
                self.tracklets[id]['last_frame'] = frame_id
                self.tracklets[id]['label'] = label
                self.tracklets[id]['velocity'] = (
                    self.tracklets[id]['velocity'] *
                    self.tracklets[id]['acc_frame'] + velocity) / (
                        self.tracklets[id]['acc_frame'] + 1)
                self.tracklets[id]['acc_frame'] += 1
            else:
                self.tracklets[id] = dict(
                    bbox=bbox,
                    mask_pred_a=mask_pred_a,
                    mask_pred_v=mask_pred_v,
                    embed=embed,
                    label=label,
                    last_frame=frame_id,
                    velocity=torch.zeros_like(bbox),
                    acc_frame=0)

        backdrop_inds = torch.nonzero(ids == -1, as_tuple=False).squeeze(1)
        ious = bbox_overlaps(bboxes[backdrop_inds, :-1], bboxes[:, :-1])
        for i, ind in enumerate(backdrop_inds):
            if (ious[i, :ind] > self.nms_backdrop_iou_thr).any():
                backdrop_inds[i] = -1
        backdrop_inds = backdrop_inds[backdrop_inds > -1]

        self.backdrops.insert(
            0,
            dict(
                bboxes=bboxes[backdrop_inds],
                mask_preds_a=mask_preds_a[backdrop_inds],
                mask_preds_v=mask_preds_v[backdrop_inds],
                embeds=embeds[backdrop_inds],
                labels=labels[backdrop_inds]))

        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            if frame_id - v['last_frame'] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

        if len(self.backdrops) > self.memo_backdrop_frames:
            self.backdrops.pop()

    @property
    def memo(self):
        memo_embeds = []
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        memo_mask_preds_a = []
        memo_mask_preds_v = []
        memo_vs = []
        for k, v in self.tracklets.items():
            memo_bboxes.append(v['bbox'][None, :])
            memo_mask_preds_a.append(v['mask_pred_a'][None, :])
            memo_mask_preds_v.append(v['mask_pred_v'][None, :])
            memo_embeds.append(v['embed'][None, :])
            memo_ids.append(k)
            memo_labels.append(v['label'].view(1, 1))
            memo_vs.append(v['velocity'][None, :])
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)

        for backdrop in self.backdrops:
            backdrop_ids = torch.full((1, backdrop['embeds'].size(0)),
                                      -1,
                                      dtype=torch.long)
            backdrop_vs = torch.zeros_like(backdrop['bboxes'])
            memo_bboxes.append(backdrop['bboxes'])
            memo_mask_preds_a.append(backdrop['mask_preds_a'])
            memo_mask_preds_v.append(backdrop['mask_preds_v'])
            memo_embeds.append(backdrop['embeds'])
            memo_ids = torch.cat([memo_ids, backdrop_ids], dim=1)
            memo_labels.append(backdrop['labels'][:, None])
            memo_vs.append(backdrop_vs)

        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_mask_preds_a = torch.cat(memo_mask_preds_a, dim=0)
        memo_mask_preds_v = torch.cat(memo_mask_preds_v, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        memo_vs = torch.cat(memo_vs, dim=0)
        return (
            memo_bboxes, memo_labels, memo_mask_preds_a, memo_mask_preds_v, memo_embeds,
            memo_ids.squeeze(0), memo_vs)

    def match(self, bboxes, labels, masks_a, masks_v, track_feats, frame_id, asso_tau=-1):

        _, inds = bboxes[:, -1].sort(descending=True)
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        mask_preds_a = masks_a['mask_pred'][inds, :]
        mask_preds_v = masks_v['mask_pred'][inds, :]

        del masks_v, masks_a
        torch.cuda.empty_cache()

        embeds = track_feats[inds, :]

        # duplicate removal for potential backdrops and cross classes
        valids = bboxes.new_ones((bboxes.size(0)))
        ious = bbox_overlaps(bboxes[:, :-1], bboxes[:, :-1])
        for i in range(1, bboxes.size(0)):
            thr = self.nms_backdrop_iou_thr if bboxes[
                i, -1] < self.obj_score_thr else self.nms_class_iou_thr
            if (ious[i, :i] > thr).any():
                valids[i] = 0
        valids = valids == 1
        bboxes = bboxes[valids, :]
        labels = labels[valids]
        mask_preds_a = mask_preds_a[valids, :]
        mask_preds_v = mask_preds_v[valids, :]
        embeds = embeds[valids, :]

        # init ids container
        ids = torch.full((bboxes.size(0), ), -1, dtype=torch.long)

        # match if buffer is not empty
        if bboxes.size(0) > 0 and not self.empty:
            (memo_bboxes, memo_labels, memo_mask_preds_a, memo_mask_preds_v, memo_embeds, memo_ids,
             memo_vs) = self.memo

            if self.match_metric == 'bisoftmax':
                feats = torch.mm(embeds, memo_embeds.t())
                d2t_scores = feats.softmax(dim=1)
                t2d_scores = feats.softmax(dim=0)
                scores = (d2t_scores + t2d_scores) / 2
            elif self.match_metric == 'softmax':
                feats = torch.mm(embeds, memo_embeds.t())
                scores = feats.softmax(dim=1)
            elif self.match_metric == 'cosine':
                scores = torch.mm(
                    F.normalize(embeds, p=2, dim=1),
                    F.normalize(memo_embeds, p=2, dim=1).t())
            else:
                raise NotImplementedError

            if self.with_cats:
                cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                cat_same = cat_same.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                scores *= cat_same.float()

            for i in range(bboxes.size(0)):
                conf, memo_ind = torch.max(scores[i, :], dim=0)
                id = memo_ids[memo_ind]
                if conf > self.match_score_thr:
                    if id > -1:
                        if bboxes[i, -1] > self.obj_score_thr:
                            ids[i] = id
                            scores[:i, memo_ind] = 0
                            scores[i + 1:, memo_ind] = 0
                        else:
                            if conf > self.nms_conf_thr:
                                ids[i] = -2
        new_inds = (ids == -1) & (bboxes[:, 4] > self.init_score_thr).cpu()
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracklets,
            self.num_tracklets + num_news,
            dtype=torch.long)
        self.num_tracklets += num_news

        self.update_memo(ids, bboxes, mask_preds_a, mask_preds_v, embeds, labels, frame_id)
        masks_a = dict(mask_pred=mask_preds_a)
        masks_v = dict(mask_pred=mask_preds_v)
        del mask_preds_v, mask_preds_a
        torch.cuda.empty_cache()

        return bboxes, labels, masks_a, masks_v, ids