import os
import sys
import json

json_file = "./joint_one_video_overtrain.json"
with open(json_file, 'rb') as f:
    json_data = json.load(f)

    annotations = json_data["annotations"]
    for item in annotations:
        item["segmentation"] = [item["segmentation"], item["visible_mask"]]
        item["amodal_mask"] = item["segmentation"][0]
        item["area_amodal"] = item["area"]
        item.pop("area", None)

with open('./new_joint_one_video_overtrain(bbox=amdl_bbox).json', 'w') as r:
    json.dump(json_data, r)