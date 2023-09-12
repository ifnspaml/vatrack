import os
import sys
import json

json_file = "./png_amodal/valid_amodal.json"
with open(json_file, 'rb') as f:
    json_data = json.load(f)

    annotations = json_data["annotations"]
    for item in annotations:
        item["segmentation"] = [item["segmentation"], item["segm_visible"]]
        item['amodal_mask'] = item['segmentation'][0]
        item["visible_mask"] = item["segm_visible"]
        item["area_amodal"] = item["area"]
        item.pop("bg_object_segmentation", None)
        item.pop("segm_visible", None)
        item.pop("area", None)

with open('./valid_joint(bbox=amdl_bbox).json', 'w') as r:
    json.dump(json_data, r)