from mmdet.core import bbox2result

from vatrack.core import segtrack2result
from ..builder import MODELS
from .quasi_dense import random_color
from .quasi_dense_pcan import EMQuasiDenseFasterRCNN

import torch
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

@MODELS.register_module()
class JointBBoxMaskQuasiDenseMaskRCNN(EMQuasiDenseFasterRCNN):

    def __init__(self, fixed=False, *args, **kwargs):
        super().__init__(channels=256, proto_num=30, stage_num=3, *args, **kwargs)  # init the pcan parameters,
        if fixed:                                                                   # no use for qdtrack
            self.fix_modules()

    def fix_modules(self):
        fixed_modules = [
            self.backbone,
            self.neck,
            self.rpn_head,
            self.roi_head.bbox_roi_extractor,
            self.roi_head.bbox_head,
            self.roi_head.track_roi_extractor,
            self.roi_head.track_head]
        for module in fixed_modules:
            # print('fixed ======================')
            for name, param in module.named_parameters():
                param.requires_grad = False

    def forward_test(self, img, img_metas, rescale=False):
        # TODO inherit from a base tracker
        assert self.roi_head.with_track, 'Track head must be implemented.'
        img_metas = img_metas[0]
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.init_tracker()

        torch.cuda.empty_cache()
        x = self.extract_feat(img[0])

        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        det_bboxes, det_labels, det_masks, track_feats = (
            self.roi_head.simple_test(x, img_metas, proposal_list, rescale))

        det_bboxes_a = det_bboxes[0]
        det_bboxes_v = det_bboxes[1]
        det_masks_a = det_masks[0]
        det_masks_v = det_masks[1]
        det_labels = det_labels[0]

        if det_bboxes_a.shape[0] != det_labels.shape[0]:
            print('det_bboxes_a.shape[0] != det_labels.shape[0]')

        bbox_result = bbox2result(det_bboxes_a, det_labels,
                                  self.roi_head.bbox_head.num_classes)

        segm_result = [
            [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        ]  # only track res are evaluated in eval, no use of segm_result, here only maintain its structure

        if track_feats is None:
            from collections import defaultdict
            track_result = defaultdict(list)
        else:
            bboxes, bboxes_v, labels, masks_a, masks_v, ids = self.tracker.match(
                bboxes=det_bboxes_a,      
                bboxes_v=det_bboxes_v,
                labels=det_labels,
                masks_a=det_masks_a,      # bbox/segm res are not exactly the same as the bbox/mask in track res.
                masks_v=det_masks_v,      # Only the det_bbox/det_mask, that are matched in track head will be then
                track_feats=track_feats,  # output in track res.
                frame_id=frame_id)        ## note: here track feat is the feat of the detected amodal bboxes

            del x, det_bboxes, det_masks_a, det_masks_v, track_feats
            torch.cuda.empty_cache()

            _, segms_a, _ = self.roi_head.get_seg_amodal_masks(
                img_metas, bboxes, labels, masks_a, rescale=rescale)
            _, segms_v, _ = self.roi_head.get_seg_masks(
                img_metas, bboxes_v, labels, masks_v, rescale=rescale)

            del masks_v, masks_a
            torch.cuda.empty_cache()

            # print(np.array_equal(segms_a[0], segms_v[0], equal_nan=True))

            track_result_a = segtrack2result(bboxes, labels, segms_a, ids)
            track_result_v = segtrack2result(bboxes_v, labels, segms_v, ids)
            track_result = [track_result_a, track_result_v]

        return dict(bbox_result=bbox_result, segm_result=segm_result,  # both segm & track res are lists
                    track_result=track_result)

    def show_result(self,
                    img,
                    result,
                    show=False,
                    out_file=None,
                    score_thr=None,
                    draw_track=True):

        track_result_a = result['track_result'][0]
        track_result_v = result['track_result'][1]

        out_file_split_a = out_file.split("/")
        out_file_split_v = out_file.split("/")
        out_file_split_a.insert(-3, 'amodal')
        out_file_split_v.insert(-3, 'visible')
        out_file_a = ("/").join(out_file_split_a)
        out_file_v = ("/").join(out_file_split_v)

        img_a = mmcv.bgr2rgb(img)
        img_b = mmcv.bgr2rgb(img)
        img_a = mmcv.imread(img_a)
        img_v = mmcv.imread(img_b)

        # visible visulization
        if not isinstance(track_result_v, list):
            for id, item in track_result_v.items():
                bbox = item['bbox']
                if bbox[-1] <= score_thr:
                    continue
                color = (np.array(random_color(id)) * 256).astype(np.uint8)
                mask_v = item['segm']
                img_v[mask_v] = img_v[mask_v] * 0.5 + color * 0.5

            plt.imshow(img_v)
            plt.gca().set_axis_off()
            plt.autoscale(False)
            plt.subplots_adjust(
                top=1, bottom=0, right=1, left=0, hspace=None, wspace=None)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            for id, item in track_result_v.items():
                bbox = item['bbox']
                if bbox[-1] <= score_thr:
                    continue
                bbox_int = bbox.astype(np.int32)
                left_top = (bbox_int[0], bbox_int[1])
                w = bbox_int[2] - bbox_int[0] + 1
                h = bbox_int[3] - bbox_int[1] + 1
                color = random_color(id)
                plt.gca().add_patch(
                    Rectangle(left_top, w, h, edgecolor=color, facecolor='none', linewidth=2))
                label_text = 'ins_id:{} cls:{} scr:{:.2}'.format(int(id), self.CLASSES[int(item['label'])],
                                                                 item['bbox'].tolist()[-1])
                bg_height = 12
                bg_width = 7
                bg_width = len(label_text) * bg_width
                if left_top[1] - bg_height > 0 and left_top[0] + bg_width < img_v.shape[1]:
                    plt.gca().add_patch(
                        Rectangle((left_top[0], left_top[1] - bg_height),
                                  bg_width,
                                  bg_height,
                                  edgecolor=color,
                                  facecolor=color))
                    plt.text(left_top[0], left_top[1], label_text, fontsize=5)
                elif left_top[1] - bg_height <= 0 and left_top[0] + bg_width < img_v.shape[1]:
                    plt.gca().add_patch(
                        Rectangle((left_top[0], left_top[1]),
                                  bg_width,
                                  bg_height,
                                  edgecolor=color,
                                  facecolor=color))
                    plt.text(left_top[0], left_top[1] + bg_height, label_text, fontsize=5)
                elif left_top[1] - bg_height > 0 and left_top[0] + bg_width >= img_v.shape[1]:
                    plt.gca().add_patch(
                        Rectangle((img_v.shape[1] - bg_width, left_top[1] - bg_height),
                                  bg_width,
                                  bg_height,
                                  edgecolor=color,
                                  facecolor=color))
                    plt.text(img_v.shape[1] - bg_width, left_top[1], label_text, fontsize=5)
                else:
                    plt.gca().add_patch(
                        Rectangle((img_v.shape[1] - bg_width, left_top[1]),
                                  bg_width,
                                  bg_height,
                                  edgecolor=color,
                                  facecolor=color))
                    plt.text(img_v.shape[1] - bg_width, left_top[1] + bg_height, label_text, fontsize=5)

            if out_file_a is not None:
                mmcv.imwrite(img_v, out_file_v)
                plt.savefig(out_file_v, dpi=300, bbox_inches='tight', pad_inches=0.0)
            plt.clf()
            plt.close('all')
        else:
            if out_file_a is not None:
                plt.imshow(img_v)
                mmcv.imwrite(img_v, out_file_v)
                plt.savefig(out_file_v, dpi=300, bbox_inches='tight', pad_inches=0.0)
            plt.clf()

        # amodal visulization
        if not isinstance(track_result_a, list):
            for id, item in track_result_a.items():
                bbox = item['bbox']
                if bbox[-1] <= score_thr:
                    continue
                color = (np.array(random_color(id)) * 256).astype(np.uint8)
                mask_a = item['segm']
                img_a[mask_a] = img_a[mask_a] * 0.5 + color * 0.5

            plt.imshow(img_a)
            plt.gca().set_axis_off()
            plt.autoscale(False)
            plt.subplots_adjust(
                top=1, bottom=0, right=1, left=0, hspace=None, wspace=None)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            for id, item in track_result_a.items():
                bbox = item['bbox']
                if bbox[-1] <= score_thr:
                    continue
                bbox_int = bbox.astype(np.int32)
                left_top = (bbox_int[0], bbox_int[1])
                w = bbox_int[2] - bbox_int[0] + 1
                h = bbox_int[3] - bbox_int[1] + 1
                color = random_color(id)
                plt.gca().add_patch(
                    Rectangle(left_top, w, h, edgecolor=color, facecolor='none', linewidth=2))
                label_text = 'ins_id:{} cls:{} scr:{:.2}'.format(int(id), self.CLASSES[int(item['label'])],
                                                                 item['bbox'].tolist()[-1])
                bg_height = 12
                bg_width = 7
                bg_width = len(label_text) * bg_width
                if left_top[1] - bg_height > 0 and left_top[0] + bg_width < img_v.shape[1]:
                    plt.gca().add_patch(
                        Rectangle((left_top[0], left_top[1] - bg_height),
                                  bg_width,
                                  bg_height,
                                  edgecolor=color,
                                  facecolor=color))
                    plt.text(left_top[0], left_top[1], label_text, fontsize=5)
                elif left_top[1] - bg_height <= 0 and left_top[0] + bg_width < img_v.shape[1]:
                    plt.gca().add_patch(
                        Rectangle((left_top[0], left_top[1]),
                                  bg_width,
                                  bg_height,
                                  edgecolor=color,
                                  facecolor=color))
                    plt.text(left_top[0], left_top[1] + bg_height, label_text, fontsize=5)
                elif left_top[1] - bg_height > 0 and left_top[0] + bg_width >= img_v.shape[1]:
                    plt.gca().add_patch(
                        Rectangle((img_v.shape[1] - bg_width, left_top[1] - bg_height),
                                  bg_width,
                                  bg_height,
                                  edgecolor=color,
                                  facecolor=color))
                    plt.text(img_v.shape[1] - bg_width, left_top[1], label_text, fontsize=5)
                else:
                    plt.gca().add_patch(
                        Rectangle((img_v.shape[1] - bg_width, left_top[1]),
                                  bg_width,
                                  bg_height,
                                  edgecolor=color,
                                  facecolor=color))
                    plt.text(img_v.shape[1] - bg_width, left_top[1] + bg_height, label_text, fontsize=5)

            if out_file_a is not None:
                mmcv.imwrite(img_a, out_file_a)
                plt.savefig(out_file_a, dpi=300, bbox_inches='tight', pad_inches=0.0)
            plt.clf()
        else:
            if out_file_a is not None:
                plt.imshow(img_a)
                mmcv.imwrite(img_a, out_file_a)
                plt.savefig(out_file_a, dpi=300, bbox_inches='tight', pad_inches=0.0)
            plt.clf()

        return img_a, img_v