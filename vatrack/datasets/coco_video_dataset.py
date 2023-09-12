import random
import time

import mmcv
import numpy as np
from mmdet.datasets import DATASETS, CocoDataset
from vatrack.core import eval_mot, eval_mots

from .parsers import CocoVID


@DATASETS.register_module()
class CocoVideoDataset(CocoDataset):

    CLASSES = None

    def __init__(self,
                 load_as_video=True,
                 match_gts=True,
                 skip_nomatch_pairs=True,
                 key_img_sampler=dict(interval=1),
                 ref_img_sampler=dict(
                     scope=3, num_ref_imgs=1, method='uniform'),
                 *args,
                 **kwargs):
        self.load_as_video = load_as_video
        self.match_gts = match_gts
        self.skip_nomatch_pairs = skip_nomatch_pairs
        self.key_img_sampler = key_img_sampler
        self.ref_img_sampler = ref_img_sampler
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        if not self.load_as_video:
            data_infos = super().load_annotations(ann_file)
        else:
            data_infos = self.load_video_anns(ann_file)
        return data_infos

    def load_video_anns(self, ann_file):
        self.coco = CocoVID(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        data_infos = []
        self.vid_ids = self.coco.get_vid_ids()
        self.img_ids = []
        for vid_id in self.vid_ids:
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            img_ids = self.key_img_sampling(img_ids, **self.key_img_sampler)
            self.img_ids.extend(img_ids)
            for img_id in img_ids:
                info = self.coco.load_imgs([img_id])[0]
                if len(info['file_name'].split('/')) > 2:
                    replace_token = info['file_name'].split('/')[0] + '/' + info['file_name'].split('/')[1] + '/'
                    info['file_name'] = info['file_name'].replace(replace_token, info['file_name'].split('/')[0] + '/')
                info['filename'] = info['file_name']
                data_infos.append(info)
        return data_infos

    def key_img_sampling(self, img_ids, interval=1):
        return img_ids[::interval]

    def ref_img_sampling(self,
                         img_info,
                         scope,
                         num_ref_imgs=1,
                         method='uniform'):
        if num_ref_imgs != 1 or method != 'uniform':
            raise NotImplementedError
        if img_info.get('frame_id', -1) < 0 or scope <= 0:
            ref_img_info = img_info.copy()
        else:
            vid_id = img_info['video_id']
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            frame_id = img_info['frame_id']
            if method == 'uniform':
                left = max(0, frame_id - scope)
                right = min(frame_id + scope, len(img_ids) - 1)
                valid_inds = img_ids[left:frame_id] + img_ids[frame_id +
                                                              1:right + 1]
                ref_img_id = random.choice(valid_inds)
            ref_img_info = self.coco.loadImgs([ref_img_id])[0]
            ref_img_info['filename'] = ref_img_info['file_name']
        return ref_img_info

    def ref_img_sampling_test(self,
                         img_info,
                         num_ref_imgs=1,
                         method='uniform'):
        if num_ref_imgs != 1 or method != 'uniform':
            raise NotImplementedError

        if img_info.get('frame_id', -1) <= 0:
            ref_img_info = img_info.copy()
        else:
            vid_id = img_info['video_id']
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            frame_id = img_info['frame_id']
            '''
            if method == 'uniform':
                left = max(0, frame_id - scope)
                right = min(frame_id + scope, len(img_ids) - 1)
                valid_inds = img_ids[left:frame_id] + img_ids[frame_id +
                                                              1:right + 1]
                ref_img_id = random.choice(valid_inds)
            '''
            ref_img_id = img_ids[frame_id - 1]
            ref_img_info = self.coco.loadImgs([ref_img_id])[0]
            ref_img_info['filename'] = ref_img_info['file_name']
        return ref_img_info

    def _pre_pipeline(self, _results):
        super().pre_pipeline(_results)
        _results['frame_id'] = _results['img_info'].get('frame_id', -1)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        if isinstance(results, list):
            for _results in results:
                self._pre_pipeline(_results)
        elif isinstance(results, dict):
            self._pre_pipeline(results)
        else:
            raise TypeError('input must be a list or a dict')

    def get_ann_info(self, img_info):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_id = img_info['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id], cat_ids=self.cat_ids)
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(img_info, ann_info)

    def prepare_results(self, img_info):
        ann_info = self.get_ann_info(img_info)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            idx = self.img_ids.index(img_info['id'])
            results['proposals'] = self.proposals[idx]
        return results

    def match_results(self, results, ref_results):
        match_indices, ref_match_indices = self._match_gts(
            results['ann_info'], ref_results['ann_info'])
        results['ann_info']['match_indices'] = match_indices
        ref_results['ann_info']['match_indices'] = ref_match_indices
        return results, ref_results

    def _match_gts(self, ann, ref_ann):
        if 'instance_ids' in ann:
            ins_ids = list(ann['instance_ids'])
            ref_ins_ids = list(ref_ann['instance_ids'])
            match_indices = np.array([
                ref_ins_ids.index(i) if i in ref_ins_ids else -1
                for i in ins_ids
            ])
            ref_match_indices = np.array([
                ins_ids.index(i) if i in ins_ids else -1 for i in ref_ins_ids
            ])
        else:
            match_indices = np.arange(ann['bboxes'].shape[0], dtype=np.int64)
            ref_match_indices = match_indices.copy()
        return match_indices, ref_match_indices

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        img_info = self.data_infos[idx]
        # print('img info:', img_info)
        ref_img_info = self.ref_img_sampling(img_info, **self.ref_img_sampler)

        results = self.prepare_results(img_info)
        ref_results = self.prepare_results(ref_img_info)

        if self.match_gts:
            results, ref_results = self.match_results(results, ref_results)
            nomatch = (results['ann_info']['match_indices'] == -1).all()
            if self.skip_nomatch_pairs and nomatch:
                return None

        self.pre_pipeline([results, ref_results])
        return self.pipeline([results, ref_results])
    
    # def prepare_test_img(self, idx):
    #     """Get training data and annotations after pipeline.

    #     Args:
    #         idx (int): Index of data.

    #     Returns:
    #         dict: Training data and annotation after pipeline with new keys \
    #             introduced by pipeline.
    #     """
    #     img_info = self.data_infos[idx]
    #     results = dict(img_info=img_info)
    #     ref_img_info = self.ref_img_sampling_test(img_info)
    #     ref_results = dict(img_info=ref_img_info)

    #     self.pre_pipeline([results, ref_results])
    #     return self.pipeline([results, ref_results])

    def _parse_ann_info(self, img_info, ann_info):  # _parse_ann_info for joint training of bbox/mask heads
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        if ann_info:
            if len(ann_info[0]['bbox']) != 4:
                gt_bboxes_a = []
                gt_bboxes_v = []
                gt_labels = []
                gt_bboxes_ignore_a = []
                gt_bboxes_ignore_v = []
                gt_masks_ann = []
                gt_instance_ids = []

                for i, ann in enumerate(ann_info):
                    if ann.get('ignore', False):
                        continue
                    x1_a, y1_a, w_a, h_a = ann['bbox'][0]
                    x1_v, y1_v, w_v, h_v = ann['bbox'][1]
                    inter_w_a = max(0, min(x1_a + w_a, img_info['width']) - max(x1_a, 0))
                    inter_h_a = max(0, min(y1_a + h_a, img_info['height']) - max(y1_a, 0))
                    inter_w_v = max(0, min(x1_v + w_v, img_info['width']) - max(x1_v, 0))
                    inter_h_v = max(0, min(y1_v + h_v, img_info['height']) - max(y1_v, 0))
                    if inter_w_a * inter_h_a == 0 or inter_w_v * inter_h_v == 0:
                        continue
                    if ann['area_visible'] <= 0 or w_a < 1 or h_a < 1 or w_v < 1 or h_v < 1:
                        continue
                    if ann['category_id'] not in self.cat_ids:
                        continue
                    bbox_a = [x1_a, y1_a, x1_a + w_a, y1_a + h_a]
                    bbox_v = [x1_v, y1_v, x1_v + w_v, y1_v + h_v]
                    if ann.get('iscrowd', False):
                        gt_bboxes_ignore_a.append(bbox_a)
                        gt_bboxes_ignore_v.append(bbox_v)
                    else:
                        gt_bboxes_a.append(bbox_a)
                        gt_bboxes_v.append(bbox_v)
                        gt_labels.append(self.cat2label[ann['category_id']])
                        if ann.get('segmentation', False):
                            gt_masks_ann.append(ann['segmentation'])
                        instance_id = ann.get('instance_id', None)
                        if instance_id is not None:
                            gt_instance_ids.append(ann['instance_id'])

                if gt_bboxes_a:
                    gt_bboxes_a = np.array(gt_bboxes_a, dtype=np.float32)
                    gt_labels = np.array(gt_labels, dtype=np.int64)
                else:
                    gt_bboxes_a = np.zeros((0, 4), dtype=np.float32)
                    gt_labels = np.array([], dtype=np.int64)

                if gt_bboxes_v:
                    gt_bboxes_v = np.array(gt_bboxes_v, dtype=np.float32)
                    gt_labels = np.array(gt_labels, dtype=np.int64)
                else:
                    gt_bboxes_v = np.zeros((0, 4), dtype=np.float32)
                    gt_labels = np.array([], dtype=np.int64)

                if gt_bboxes_ignore_a:
                    gt_bboxes_ignore_a = np.array(gt_bboxes_ignore_a, dtype=np.float32)
                else:
                    gt_bboxes_ignore_a = np.zeros((0, 4), dtype=np.float32)

                if gt_bboxes_ignore_v:
                    gt_bboxes_ignore_v = np.array(gt_bboxes_ignore_v, dtype=np.float32)
                else:
                    gt_bboxes_ignore_v = np.zeros((0, 4), dtype=np.float32)

                seg_map = img_info['filename'].replace('jpg', 'png')

                bboxes_output = [gt_bboxes_a, gt_bboxes_v]
                if gt_bboxes_ignore_a:
                    gt_bboxes_ignore_output = [gt_bboxes_ignore_a, gt_bboxes_ignore_v]
                else:
                    gt_bboxes_ignore_output = np.zeros((0, 4), dtype=np.float32)

                ann = dict(
                    bboxes=bboxes_output,
                    labels=gt_labels,
                    bboxes_ignore=gt_bboxes_ignore_output,
                    masks=gt_masks_ann,
                    seg_map=seg_map)

                if self.load_as_video:
                    ann['instance_ids'] = np.array(gt_instance_ids).astype(np.int)
                else:
                    ann['instance_ids'] = np.arange(len(gt_labels))

                return ann

            else:
                gt_bboxes = []
                gt_labels = []
                gt_bboxes_ignore = []
                gt_masks_ann = []
                gt_instance_ids = []

                for i, ann in enumerate(ann_info):
                    if ann.get('ignore', False):
                        continue
                    x1, y1, w, h = ann['bbox']
                    inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
                    inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
                    if inter_w * inter_h == 0:
                        continue
                    if ann['area_visible'] <= 0 or w < 1 or h < 1:
                        continue
                    if ann['category_id'] not in self.cat_ids:
                        continue
                    bbox = [x1, y1, x1 + w, y1 + h]
                    if ann.get('iscrowd', False):
                        gt_bboxes_ignore.append(bbox)
                    else:
                        gt_bboxes.append(bbox)
                        gt_labels.append(self.cat2label[ann['category_id']])
                        if ann.get('segmentation', False):
                            gt_masks_ann.append(ann['segmentation'])
                        instance_id = ann.get('instance_id', None)
                        if instance_id is not None:
                            gt_instance_ids.append(ann['instance_id'])

                if gt_bboxes:
                    gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                    gt_labels = np.array(gt_labels, dtype=np.int64)
                else:
                    gt_bboxes = np.zeros((0, 4), dtype=np.float32)
                    gt_labels = np.array([], dtype=np.int64)

                if gt_bboxes_ignore:
                    gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
                else:
                    gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

                seg_map = img_info['filename'].replace('jpg', 'png')

                ann = dict(
                    bboxes=gt_bboxes,
                    labels=gt_labels,
                    bboxes_ignore=gt_bboxes_ignore,
                    masks=gt_masks_ann,
                    seg_map=seg_map)

                if self.load_as_video:
                    ann['instance_ids'] = np.array(gt_instance_ids).astype(np.int)
                else:
                    ann['instance_ids'] = np.arange(len(gt_labels))

                return ann

        else:
            gt_bboxes_a = []
            gt_bboxes_v = []
            gt_labels = []
            gt_bboxes_ignore_a = []
            gt_bboxes_ignore_v = []
            gt_masks_ann = []
            gt_instance_ids = []

            for i, ann in enumerate(ann_info):
                if ann.get('ignore', False):
                    continue
                x1_a, y1_a, w_a, h_a = ann['bbox'][0]
                x1_v, y1_v, w_v, h_v = ann['bbox'][1]
                inter_w_a = max(0, min(x1_a + w_a, img_info['width']) - max(x1_a, 0))
                inter_h_a = max(0, min(y1_a + h_a, img_info['height']) - max(y1_a, 0))
                inter_w_v = max(0, min(x1_v + w_v, img_info['width']) - max(x1_v, 0))
                inter_h_v = max(0, min(y1_v + h_v, img_info['height']) - max(y1_v, 0))
                if inter_w_a * inter_h_a == 0 or inter_w_v * inter_h_v == 0:
                    continue
                if ann['area_visible'] <= 0 or w_a < 1 or h_a < 1 or w_v < 1 or h_v < 1:
                    continue
                if ann['category_id'] not in self.cat_ids:
                    continue
                bbox_a = [x1_a, y1_a, x1_a + w_a, y1_a + h_a]
                bbox_v = [x1_v, y1_v, x1_v + w_v, y1_v + h_v]
                if ann.get('iscrowd', False):
                    gt_bboxes_ignore_a.append(bbox_a)
                    gt_bboxes_ignore_v.append(bbox_v)
                else:
                    gt_bboxes_a.append(bbox_a)
                    gt_bboxes_v.append(bbox_v)
                    gt_labels.append(self.cat2label[ann['category_id']])
                    if ann.get('segmentation', False):
                        gt_masks_ann.append(ann['segmentation'])
                    instance_id = ann.get('instance_id', None)
                    if instance_id is not None:
                        gt_instance_ids.append(ann['instance_id'])

            if gt_bboxes_a:
                gt_bboxes_a = np.array(gt_bboxes_a, dtype=np.float32)
                gt_labels = np.array(gt_labels, dtype=np.int64)
            else:
                gt_bboxes_a = np.zeros((0, 4), dtype=np.float32)
                gt_labels = np.array([], dtype=np.int64)

            if gt_bboxes_v:
                gt_bboxes_v = np.array(gt_bboxes_v, dtype=np.float32)
                gt_labels = np.array(gt_labels, dtype=np.int64)
            else:
                gt_bboxes_v = np.zeros((0, 4), dtype=np.float32)
                gt_labels = np.array([], dtype=np.int64)

            if gt_bboxes_ignore_a:
                gt_bboxes_ignore_a = np.array(gt_bboxes_ignore_a, dtype=np.float32)
            else:
                gt_bboxes_ignore_a = np.zeros((0, 4), dtype=np.float32)

            if gt_bboxes_ignore_v:
                gt_bboxes_ignore_v = np.array(gt_bboxes_ignore_v, dtype=np.float32)
            else:
                gt_bboxes_ignore_v = np.zeros((0, 4), dtype=np.float32)

            seg_map = img_info['filename'].replace('jpg', 'png')

            bboxes_output = [gt_bboxes_a, gt_bboxes_v]
            if gt_bboxes_ignore_a:
                gt_bboxes_ignore_output = [gt_bboxes_ignore_a, gt_bboxes_ignore_v]
            else:
                gt_bboxes_ignore_output = np.zeros((0, 4), dtype=np.float32)

            ann = dict(
                bboxes=bboxes_output,
                labels=gt_labels,
                bboxes_ignore=gt_bboxes_ignore_output,
                masks=gt_masks_ann,
                seg_map=seg_map)

            if self.load_as_video:
                ann['instance_ids'] = np.array(gt_instance_ids).astype(np.int)
            else:
                ann['instance_ids'] = np.arange(len(gt_labels))

            return ann

    def format_track_results(self, results, **kwargs):
        pass

    def evaluate(self,
                 results,
                 metric=['bbox', 'segm', 'track', 'segtrack'],
                 logger=None,
                 classwise=True,
                 mot_class_average=True,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=None,
                 metric_items=None):
        # evaluate for detectors without tracker
        #mot_class_average=False
        mot_class_average=True
        eval_results = dict()
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'track', 'segtrack']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        res_total = {}

        #### transfer joint res into amodal/vis res
        track_result_a = [it[0] for it in results['track_result']]
        track_result_v = [it[1] for it in results['track_result']]
        segm_result_no_real_use = [[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
                                   for _ in range(len(results['track_result']))]   # only inherit the structure, not the data

        # modal frame-level evaluation
        # for SAIL-VOScut modal frame-level evaluation
        if "cut" in self.ann_file:
            print('\n\n------------SAIL-VOScut modal frame-level evaluation------------\n')
        # for SAIL-VOS modal frame-level evaluation
        else:
            print('\n\n------------SAIL-VOS modal frame-level evaluation------------\n')

        print('\nNote: When in joint setting, BBox results are only evaluated with the amodal GT BBox. Since BBox performance is not required in this work, just ignore them and focus on Mask results.\n')
        results['track_result'] = track_result_v
        results['segm_result'] = segm_result_no_real_use
        super_metrics = ['bbox', 'segm']
        if super_metrics:
            if 'bbox' in super_metrics and 'segm' in super_metrics:
                super_results = []

                # get bbox/segm res from track results
                print("\nstart to transfer visible track res into visible bbox/segm res")
                start = time.time()
                for i in range(len(results["track_result"])):
                    for j in range(len(self.CLASSES)):
                        results["bbox_result"][i][j] = np.empty((0, 5))
                        results["segm_result"][i][j] = []
                    if results["track_result"][i].keys():
                        for item in results["track_result"][i].keys():
                            a = results["bbox_result"][i][results["track_result"][i][item]["label"]]
                            b = results["track_result"][i][item]["bbox"].reshape(1, 5)
                            results["bbox_result"][i][results["track_result"][i][item]["label"]] = np.append(a, b,
                                                                                                             axis=0)
                            segm_res = results["segm_result"][i][results["track_result"][i][item]["label"]]
                            segm_res.append(results["track_result"][i][item]["segm"])
                print("visible track res transfer cost {}s\n".format(time.time() - start))

                for bbox, segm in zip(results['bbox_result'], results['segm_result']):
                    super_results.append((bbox, segm))
            else:
                super_results = results['bbox_result']
            from .sailvos import SAILVOS
            super_eval_results = SAILVOS.evaluate(
                self,
                results=super_results,
                metric=super_metrics,
                logger=logger,
                classwise=classwise,
                proposal_nums=proposal_nums,
                iou_thrs=iou_thr,
                metric_items=metric_items,
                a_or_v="v")
            eval_results.update(super_eval_results)
            res_total["visible track bbox/segm mAP"] = {}
            res_total["visible track bbox/segm mAP"] = super_eval_results


        # amodal frame-level evaluation
        # for SAIL-VOScut amodal frame-level evaluation
        if "cut" in self.ann_file:
            print('\n\n------------SAIL-VOScut amodal frame-level evaluation------------\n')
        # for SAIL-VOS amodal frame-level evaluation
        else:
            print('\n\n------------SAIL-VOS amodal frame-level evaluation------------\n')

        print('\nNote: When in joint setting, BBox results are only evaluated with the amodal GT BBox. Since BBox performance is not required in this work, just ignore them and focus on Mask results.\n')
        results['track_result'] = track_result_a
        results['segm_result'] = segm_result_no_real_use
        super_metrics = ['bbox', 'segm']
        if super_metrics:
            if 'bbox' in super_metrics and 'segm' in super_metrics:
                super_results = []

                # get bbox/segm res from track results
                print("\nstart to transfer amodal track res into amodal bbox/segm res")
                start = time.time()
                for i in range(len(results["track_result"])):
                    for j in range(len(self.CLASSES)):
                        results["bbox_result"][i][j] = np.empty((0, 5))
                        results["segm_result"][i][j] = []
                    if results["track_result"][i].keys():
                        for item in results["track_result"][i].keys():
                            a = results["bbox_result"][i][results["track_result"][i][item]["label"]]
                            b = results["track_result"][i][item]["bbox"].reshape(1, 5)
                            results["bbox_result"][i][results["track_result"][i][item]["label"]] = np.append(a, b,
                                                                                                             axis=0)
                            segm_res = results["segm_result"][i][results["track_result"][i][item]["label"]]
                            segm_res.append(results["track_result"][i][item]["segm"])
                print("amodal track res transfer cost {}s\n".format(time.time() - start))

                for bbox, segm in zip(results['bbox_result'], results['segm_result']):
                    super_results.append((bbox, segm))
            else:
                super_results = results['bbox_result']
            from .sailvos import SAILVOS
            super_eval_results = SAILVOS.evaluate(
                self,
                results=super_results,
                metric=super_metrics,
                logger=logger,
                classwise=classwise,
                proposal_nums=proposal_nums,
                iou_thrs=iou_thr,
                metric_items=metric_items,
                a_or_v="a")
            eval_results.update(super_eval_results)
            res_total["amodal track bbox/segm mAP"] = {}
            res_total["amodal track bbox/segm mAP"] = super_eval_results


        if 'segtrack' in metrics:

            # amodal MOTS results, can be omitted
            print("\n\nstart calculating amodal MOTS results\n")
            results['track_result'] = track_result_v
            anns = mmcv.load(self.ann_file)
            if len(anns['annotations'][0]['bbox']) != 4:
                for item in anns['annotations']:
                    item['bbox'] = item['bbox'][0]
            track_eval_results = eval_mots(
                self.coco,
                anns,
                results['track_result'],
                class_average=mot_class_average,
                a_or_v="v")
            eval_results.update(track_eval_results)

            # visible MOTS results, can be omitted
            print("\n\nstart calculating visible MOTS results\n")
            results['track_result'] = track_result_a
            for item in anns['annotations']:
                item['bbox'] = item['bbox_visible']
            track_eval_results = eval_mots(
                self.coco,
                anns,
                results['track_result'],
                class_average=mot_class_average,
                a_or_v="a")
            eval_results.update(track_eval_results)

        elif 'track' in metrics:

            # amodal MOT results, can be omitted
            print("\n\nstart calculating amodal MOT results\n")
            results['track_result'] = track_result_v
            anns = mmcv.load(self.ann_file)
            if len(anns['annotations'][0]['bbox']) != 4:
                for item in anns['annotations']:
                    item['bbox'] = item['bbox'][0]
            track_eval_results = eval_mot(
                anns,
                results['track_result'],
                class_average=mot_class_average,
                a_or_v="v")
            eval_results.update(track_eval_results)

            # visible MOT results, can be omitted
            print("\n\nstart calculating visible MOT results\n")
            results['track_result'] = track_result_a
            for item in anns['annotations']:
                item['bbox'] = item['bbox_visible']
            track_eval_results = eval_mot(
                anns,
                results['track_result'],
                class_average=mot_class_average,
                a_or_v="a")
            eval_results.update(track_eval_results)



        ################################ evaluate the visible mAP of VIS ################################
        print("\n\nStart calculating modal video-level mAP\n")
        results['track_result'] = track_result_v
        if 'segm' in metrics:
            mAP_eval_results = []  #### per instance annoattion (with bbox / segm / score)
            res = results["track_result"]

            ## YouTube_VIS format per video results
            print("start to transfer per img visible res into per vid visible res")
            start = time.time()
            video_id = 1
            res_ytvis = []
            res_ytvis_video = []
            for i in range(len(res)):  ## firstly split the images into all videos, then instances
                if self.data_infos[i]["video_id"] == video_id:
                    res_ytvis_video.append(res[i])
                else:
                    res_ytvis.append(res_ytvis_video)
                    video_id += 1
                    res_ytvis_video = []
                    res_ytvis_video.append(res[i])
            res_ytvis.append(res_ytvis_video)  ## 989 videos(list) with 26873 per img annos

            for j in range(len(res_ytvis)):
                check_none_video = 0
                map_instance_cat = {}
                for item in res_ytvis[j]:  ## check if there is any instance in the video
                    if item.__len__() != 0:
                        check_none_video = 1
                        break
                if check_none_video == 0:  ## empty video without any instance
                    print("video {} is empty (visible VIS)".format(j + 1))
                    pass
                else:  ## non-empty video
                    instance_max = 0
                    for m in range(len(res_ytvis[j])):
                        for key in res_ytvis[j][m].keys():  # here key is the ins_id in one video
                            if key not in map_instance_cat.keys():
                                map_instance_cat[key] = int(res_ytvis[j][m][key]["label"] + 1)
                            else:
                                if map_instance_cat[key] != int(res_ytvis[j][m][key]["label"] + 1):
                                    print("ERROR: in video {}, identical ins with diff cat (visible VIS)".format(
                                        j + 1))

                            if int(key) > instance_max:
                                instance_max = int(key)

                    set_ins_id = set(map_instance_cat.keys())
                    if len(set_ins_id) != instance_max + 1:
                        print("ERROR: in video {}, the instance ids in vid are not continuous (visible VIS)".format(
                            j + 1))

                    for n in range(instance_max + 1):  ## n is the ins_id in a video
                        dict_ele = {}
                        dict_ele["video_id"] = j + 1
                        if n in map_instance_cat.keys():
                            dict_ele["category_id"] = map_instance_cat[n]
                        else:
                            print("ERROR: in video {}, instance:{} is dropped? (visible VIS)".format(j + 1, n))

                        dict_ele["iscrowd"] = 0
                        dict_ele["segmentations"] = []
                        dict_ele["bboxes"] = []
                        score = 0.0  ## use average score as the score of the instance in the video
                        score_times = 0
                        for q in range(len(res_ytvis[j])):  # q is frame_id of a video from 0
                            if n in res_ytvis[j][q].keys():  # if this ins in that frame, append infos
                                dict_ele["segmentations"].append(res_ytvis[j][q][n]["segm"])
                                list_xywh = res_ytvis[j][q][n]["bbox"].tolist()
                                x, y, w, h = list_xywh[0], list_xywh[1], list_xywh[2] - list_xywh[0], list_xywh[3] - \
                                             list_xywh[1]
                                dict_ele["bboxes"].append([x, y, w, h])
                                score += list_xywh[4]
                                score_times += 1
                            else:  # if this ins not in that frame, append None
                                dict_ele["segmentations"].append(None)
                                dict_ele["bboxes"].append(None)
                        if score_times > 0:
                            dict_ele["score"] = score / score_times  # use ave-score as it says in the paper
                            if dict_ele["score"] >= 1.0 or dict_ele["score"] <= 0.0:
                                print(
                                    "Error: video:{}, instance:{}'s score not in (0, 1) (visible VIS)".format(j + 1,
                                                                                                              n))
                        else:
                            print("Error: video:{}, instance:{} has no scores (visible VIS)".format(j + 1, n))
                        mAP_eval_results.append(dict_ele)
            print("res transfer costs {}s (visible VIS)\n".format(time.time() - start))

            ## YouTube-VIS format per video annotation
            print("\nstart inputing per video anno (visible VIS)")
            start = time.time()
            import json

            # for SAIL-VOScut modal video-level evaluation
            if "cut" in self.ann_file:
                print('\n\n------------SAIL-VOScut modal video-level evaluation------------\n')
                dataset = json.load(
                    open("../data/sailvos_cut_json/png_visible/valid_less0.75_png_visible_per_video.json", 'r'))

            # for SAIL-VOS modal video-level evaluation
            else:
                print('\n\n------------SAIL-VOS modal video-level evaluation------------\n')
                dataset = json.load(
                    open("../data/sailvos_complete_json/png_visible_cmplt_video/valid_less0.75_png_visible_cmplt_vid_per_video.json", 'r'))

            print("inputing per video anno costs {}s (visible VIS)\n".format(time.time() - start))

            from mmtrack.core.evaluation.eval_vis import eval_vis
            visible_vis_eval_results = eval_vis(
                test_results=mAP_eval_results,
                vis_anns=dataset,
                maxDets=[1, 10, 100])
            eval_results.update(visible_vis_eval_results)
            res_total["visible per video segm mAP"] = {}
            res_total["visible per video segm mAP"] = visible_vis_eval_results



        ################################ evaluate the amodal mAP of VIS ################################
        print("\n\nStart calculating amodal video-level mAP\n")
        results['track_result'] = track_result_a
        if 'segm' in metrics:
            mAP_eval_results = []  #### per instance annoattion (with bbox / segm / score)
            res = results["track_result"]

            ## YouTube_VIS format per video results
            print("start to transfer per img amodal res into per vid amodal res")
            start = time.time()
            video_id = 1
            res_ytvis = []
            res_ytvis_video = []
            for i in range(len(res)):  ## firstly split the instances into all videos
                if self.data_infos[i]["video_id"] == video_id:
                    res_ytvis_video.append(res[i])
                else:
                    res_ytvis.append(res_ytvis_video)
                    video_id += 1
                    res_ytvis_video = []
                    res_ytvis_video.append(res[i])
            res_ytvis.append(res_ytvis_video)  ## 989 videos(list) with 26873 per img annos

            for j in range(len(res_ytvis)):
                check_none_video = 0
                map_instance_cat = {}
                for item in res_ytvis[j]:  ## check if there is any instance in the video
                    if item.__len__() != 0:
                        check_none_video = 1
                        break
                if check_none_video == 0:  ## empty video without any instance
                    print("video {} is empty (amodal VIS)".format(j + 1))
                    pass
                else:  ## non-empty video
                    instance_max = 0
                    for m in range(len(res_ytvis[j])):
                        for key in res_ytvis[j][m].keys():  # here key is the ins_id in one video
                            if key not in map_instance_cat.keys():
                                map_instance_cat[key] = int(res_ytvis[j][m][key]["label"] + 1)
                            else:
                                if map_instance_cat[key] != int(res_ytvis[j][m][key]["label"] + 1):
                                    print("ERROR: in video {}, identical ins with diff cat (amodal VIS)".format(
                                        j + 1))

                            if int(key) > instance_max:
                                instance_max = int(key)

                    set_ins_id = set(map_instance_cat.keys())
                    if len(set_ins_id) != instance_max + 1:
                        print("ERROR: in video {}, the instance ids in vid are not continuous (amodal VIS)".format(
                            j + 1))

                    for n in range(instance_max + 1):  ## n is the ins_id in a video
                        dict_ele = {}
                        dict_ele["video_id"] = j + 1
                        if n in map_instance_cat.keys():
                            dict_ele["category_id"] = map_instance_cat[n]
                        else:
                            print("ERROR: in video {}, instance:{} is dropped? (amodal VIS)".format(j + 1, n))

                        dict_ele["iscrowd"] = 0
                        dict_ele["segmentations"] = []
                        dict_ele["bboxes"] = []
                        score = 0.0  ## use average score as the score of the instance in the video
                        score_times = 0
                        for q in range(len(res_ytvis[j])):  # q is frame_id of a video from 0
                            if n in res_ytvis[j][q].keys():  # if this ins in that frame, append infos
                                dict_ele["segmentations"].append(res_ytvis[j][q][n]["segm"])
                                list_xywh = res_ytvis[j][q][n]["bbox"].tolist()
                                x, y, w, h = list_xywh[0], list_xywh[1], list_xywh[2] - list_xywh[0], list_xywh[3] - \
                                             list_xywh[1]
                                dict_ele["bboxes"].append([x, y, w, h])
                                score += list_xywh[4]
                                score_times += 1
                            else:  # if this ins not in that frame, append None
                                dict_ele["segmentations"].append(None)
                                dict_ele["bboxes"].append(None)
                        if score_times > 0:
                            dict_ele["score"] = score / score_times  # use ave-score as it says in the paper
                            if dict_ele["score"] >= 1.0 or dict_ele["score"] <= 0.0:
                                print(
                                    "Error: video:{}, instance:{}'s score not in (0, 1) (amodal VIS)".format(j + 1,
                                                                                                              n))
                        else:
                            print("Error: video:{}, instance:{} has no scores (amodal VIS)".format(j + 1, n))
                        mAP_eval_results.append(dict_ele)
            print("res transfer costs {}s (amodal VIS)\n".format(time.time() - start))

            ## YouTube-VIS format per video annotation
            print("\nstart inputing per video anno (amodal VIS)")
            start = time.time()
            import json

            # for SAIL-VOScut amodal video-level evaluation
            if "cut" in self.ann_file:
                print('\n\n------------SAIL-VOScut amodal video-level evaluation------------\n')
                dataset = json.load(
                    open("../data/sailvos_cut_json/png_amodal/valid_less0.75_png_amodal_per_video.json", 'r'))

            # for SAIL-VOS amodal video-level evaluation
            else:
                print('\n\n------------SAIL-VOS amodal video-level evaluation------------\n')
                dataset = json.load(
                    open("../data/sailvos_complete_json/png_amodal_cmplt_video/valid_less0.75_png_amodal_cmplt_vid_per_video.json", 'r'))

            print("inputing per video anno costs {}s (amodal VIS)\n".format(time.time() - start))

            from mmtrack.core.evaluation.eval_vis import eval_vis
            amodal_vis_eval_results = eval_vis(
                test_results=mAP_eval_results,
                vis_anns=dataset,
                maxDets=[1, 10, 100])
            eval_results.update(amodal_vis_eval_results)
            res_total["amodal per video segm mAP"] = {}
            res_total["amodal per video segm mAP"] = amodal_vis_eval_results


        for key_res in res_total.keys():
            print("\n{} is {}\n".format(key_res, res_total[key_res]))
        return eval_results
