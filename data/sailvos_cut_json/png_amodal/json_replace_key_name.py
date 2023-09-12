import os
import sys
import json

json_file = "./valid.json"
with open(json_file, 'rb') as f:
    json_data = json.load(f)

    ## map sailvos cat_id into 1-24 cls
    # print(json_data.keys())
    # cls_mapping = {}
    # cat_list = json_data['categories']
    # n = 0
    # for item in cat_list:
    #     n += 1
    #     cls_mapping[item["id"]] = n
    #     item['id'] = n
    #
    # print("new_cat_dict: {}".format(json_data["categories"]))
    # print('cls_mapping: {}'.format(cls_mapping))
    #
    # models = json_data["models"]
    # for item in models:
    #     if item["category_id"] in cls_mapping.keys():
    #         item["category_id"] = cls_mapping[item["category_id"]]
    #         # print(item)
    #     else:
    #         print("models has cat_id out of given")
    #         pass
    #
    # annotations = json_data["annotations"]
    # for item in annotations:
    #     if item["category_id"] in cls_mapping.keys():
    #         item["category_id"] = cls_mapping[item["category_id"]]
    #     else:
    #         raise ValueError("category_id not in cls_mapping")


    ## change invisible mask into bg object segmentation
    # annotations = json_data["annotations"]
    #
    # n = 0
    # for item in annotations:
    #     n += 1
    #     item["bg_object_segmentation"] = item["invisible_mask"]
    #     print(item) if n < 3 else None
    #     item.pop("invisible_mask", None)
    #     print(item) if n < 3 else None
    #
    # replace amodal segm/bbox/area with visible segm/bbox/area
    annotations = json_data["annotations"]
    for item in annotations:
        # item["segm_amodal"], item["segmentation"] = item["segmentation"], item["segm_amodal"]
        # item["bbox_amodal"], item["bbox"] = item["bbox"], item["bbox_amodal"]
        # item["area_amodal"], item["area"] = item["area"], item["area_amodal"]
        item["segm_visible"] = item["segm_amodal"]
        item.pop("segm_amodal", None)
        item["bbox_visible"] = item["bbox_amodal"]
        item.pop("bbox_amodal", None)
        item["area_visible"] = item["area_amodal"]
        item.pop("area_amodal", None)


    anno_new = []
    for item in annotations:
        if item["area"] > 0 and item["occlude_rate"] < 0.75:
            anno_new.append(item)

    json_data["annotations"] = anno_new #annotations

    # for item in json_data["images"]:
    #     item["file_name"] = item["file_name"].replace(".bmp", ".png")

    ## overtrain
    # images_new = []
    # for item in json_data["images"]:
    #     if "ah_3a_ext__253" in item["file_name"]:
    #         images_new.append(item)
    #     else:
    #         pass
    #
    # image_ids = set()
    # for item in json_data["annotations"]:
    #     image_ids.add(item["video_id"])
    #
    # images_new = []
    # for it in json_data["images"]:
    #     if it["id"] in image_ids:
    #         images_new.append(it)
    #     else:
    #         pass
    # json_data["images"] = images_new


with open('./valid_amodal.json', 'w') as r:
    json.dump(json_data, r)