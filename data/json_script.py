import os
import sys
import json

json_file = "./ori_sailvoscut_valid.json"
with open(json_file, 'rb') as f:
    json_data = json.load(f)


    # map sailvos cat_id into 1-24 cls
    print(json_data.keys())
    cls_mapping = {}
    cat_list = json_data['categories']
    n = 0
    for item in cat_list:
        n += 1
        cls_mapping[item["id"]] = n
        item['id'] = n

    print("new_cat_dict: {}".format(json_data["categories"]))
    print('cls_mapping: {}'.format(cls_mapping))

    models = json_data["models"]
    for item in models:
        if item["category_id"] in cls_mapping.keys():
            item["category_id"] = cls_mapping[item["category_id"]]
            # print(item)
        else:
            # print("models has cat_id out of given")
            pass

    annotations = json_data["annotations"]
    for item in annotations:
        if item["category_id"] in cls_mapping.keys():
            item["category_id"] = cls_mapping[item["category_id"]]
        else:
            raise ValueError("category_id not in cls_mapping")


    # replace bmp with png
    images = json_data["images"]
    for img in images:
        img["file_name"] = img["file_name"].replace(".bmp", ".png")


    anno_new = []

    # # 1. generate new annotations for qd-based amodaltrack (amodal)
    # annotations = json_data["annotations"]
    # for item in annotations:
    #     if item["occlude_rate"] <= 0.75:
    #         anno_new.append(item)

    # # 2. generate new annotations for qdtrack (visible)
    # annotations = json_data["annotations"]
    # for item in annotations:
    #     if item["occlude_rate"] <= 0.75:
    #         item["amodal_mask"] = item['segmentation']
    #         item["segmentation"] = item["visible_mask"]
    #         item["area_amodal"] = item['area']
    #         item['area'] = item['area_visible']
    #         item['bbox_amodal'] = item['bbox']
    #         item['bbox'] = item['bbox_visible']
    #         anno_new.append(item)

    # # 3. generate new annotations for -joint model (bbox=amdl_bbox)
    # annotations = json_data["annotations"]
    # for item in annotations:
    #     if item["occlude_rate"] <= 0.75:
    #         item["amodal_mask"] = item['segmentation']
    #         item["segmentation"] = [item["amodal_mask"], item["visible_mask"]]
    #         item["area_amodal"] = item['area']
    #         del item['area']
    #         del item['invisible_mask']
    #         anno_new.append(item)

    # 4. generate new annotations for -2bbox model (bbox=both_bbox)
    annotations = json_data["annotations"]
    for item in annotations:
        if item["occlude_rate"] <= 0.75:
            item["amodal_mask"] = item['segmentation']
            item["segmentation"] = [item["amodal_mask"], item["visible_mask"]]
            item["area_amodal"] = item['area']
            del item['area']
            del item['invisible_mask']
            item['bbox_amodal'] = item['bbox']
            item['bbox'] = [item['bbox_amodal'], item['bbox_visible']]
            anno_new.append(item)


    json_data["annotations"] = anno_new




with open('xxxx.json', 'w') as r:    # save as expected name
    json.dump(json_data, r)