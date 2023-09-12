# Copyright removed for double blind review #ToDo
import argparse
import copy
import os
import os.path as osp
from collections import defaultdict

import mmcv
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='SAILVOS to COCO Video format')
    parser.add_argument(
        '-i',
        '--input',
        help='root directory of SAILVOS annotations',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save coco formatted label file',
    )

    return parser.parse_args()


def convert_vis(ann_dir, save_dir, mode='train'):
    """Convert SAILVOS dataset in COCO style.

    Args:
        ann_dir (str): The path of SAILVOS dataset.(folder)
        save_dir (str): The path to save `VIS`.
        mode (str): Convert train dataset or validation dataset or test
            dataset. Options are 'train', 'valid', 'test'. Default: 'train'.
    """

    assert mode in ['train', 'valid']

    VIS = defaultdict(list)

    # here load the json file
    official_anns = mmcv.load(osp.join(ann_dir, f'{mode}.json'))  # has to be train.json or valid.json

    print("before categories")
    VIS['categories'] = copy.deepcopy(official_anns['categories'])  # categories done

    # video id global
    print("before video")
    global_video_id = 0
    # changing the videos: adding width and height and changing vid_name to name
    for video_info in official_anns['videos']:
        global_video_id = global_video_id + 1
        video_name = video_info['vid_name']
        video = dict(
            id=global_video_id,
            name=video_name,  # here changing vid_name to name
            width=1280,  # added
            height=800,  # added
            num_frames=video_info['num_frames'],  # kept
            num_objs=video_info['num_objs'])  # kept
        print(video)
        VIS['videos'].append(video)  # videos done
        for image_info in official_anns['images']:
            first_postion = image_info['file_name'].find('/')  # find the first "/" in filename
            old_video_name = image_info['file_name'][0:first_postion]
            if video_name == old_video_name:
                image = dict(
                    id=image_info['id'],  # should not chnage!
                    video_id=global_video_id,
                    file_name=image_info['file_name'],
                    frame_id=image_info['frame_id'],
                    width=image_info['width'],
                    height=image_info['height'],
                )
                print(image)
                VIS['images'].append(image)

    # get height and width for videos and video_id for annotation
    imgToVideo = defaultdict(list)
    for img in VIS['images']:
        imgToVideo[img['id']] = img['video_id']
        # width = img['width']
        # height = img['height']
    print("imgToVideo", imgToVideo)

    # video id not  global
    """
    # get height and width for videos and video_id for annotation
    imgToVideo = defaultdict(list)
    for img in official_anns['images']:
        imgToVideo[img['id']] = img['video_id']
        width = img['width']
        height = img['height']

    print("before video")
    # changing the videos: adding width and height and changing vid_name to name
    for video_info in official_anns['videos']:
        video_name = video_info['vid_name']
        video = dict(
            id=video_info['id'],
            name=video_name,  # here changing vid_name to name
            width=width,  # added
            height=height,  # added
            num_frames=video_info['num_frames'],  # kept
            num_objs=video_info['num_objs'])  # kept
        VIS['videos'].append(video)  # videos done
    print("before images")
    VIS['images'] = copy.deepcopy(official_anns['images'])  # models done


    """

    print("before models")
    VIS['models'] = copy.deepcopy(official_anns['models'])  # models done

    print("before annotations")
    """
    all_images =0
    for imgs in official_anns['images']:
        if imgs['id'] > all_images:
            all_images = imgs['id']
        else:
            all_images = all_images
    print("all_images",all_images)
    """
    records = dict(global_instance_id=1)
    old_frames = 0
    new_frames = 0
    for vids in VIS['videos']:  # not official annot since i want to use the new ids
        vid_id = vids['id']
        object_ids_in_video = dict()  # when obj_id gets replaced with an instance id map the two values
        print("video id is ", vid_id)

        new_frames = vids['num_frames'] + new_frames

        print("old_frames are ", old_frames)
        print("new_frames are ", new_frames)

        for imageid in range(old_frames - 5, new_frames + 5):  # more than 26872 for valid#
            # changing the annotations: instance_id gets obj_id and add video key
            for anns in official_anns['annotations']:
                if anns['image_id'] == imageid and imgToVideo[
                    anns['image_id']] == vid_id:  # anns['image_id'] == frame_id and i do not  have a frame id
                    print("imageid id is ", imageid)
                    print("vid_id in loop is ", vid_id)

                    object_id = anns['obj_id']

                    if object_id in object_ids_in_video:
                        instance_id = object_ids_in_video[object_id]
                    else:
                        instance_id = records['global_instance_id']
                        records['global_instance_id'] += 1
                        object_ids_in_video[object_id] = instance_id

                    ann = dict(
                        id=anns['id'],
                        video_id=imgToVideo[anns['image_id']],
                        image_id=anns['image_id'],
                        category_id=anns['category_id'],
                        instance_id=instance_id,
                        obj_id=anns['obj_id'],
                        model_id=anns['model_id'],
                        bbox=anns['bbox'],
                        bbox_visible=anns['bbox_visible'],
                        segmentation=anns['segmentation'],
                        visible_mask=anns['visible_mask'],
                        invisible_mask=anns['invisible_mask'],
                        area=anns['area'],
                        area_visible=anns['area_visible'],
                        iscrowd=anns['iscrowd'],
                        occlu_depth=anns['occlu_depth'],
                        occlude_rate=anns['occlude_rate'])
                    VIS['annotations'].append(ann)  # videos done
                else:
                    zero = 0
        print("object_ids_in_video", object_ids_in_video)
        old_frames = new_frames

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    mmcv.dump(VIS,
              osp.join(save_dir, f'{mode}.json'))
    print(f'-----SAILVOS  {mode}------')

    print('-----------------------')

    for i in range(1, len(VIS['categories']) + 1):
        class_name = VIS['categories'][i - 1]['name']
        print(f'Class {i} {class_name} has objects.')


def main():
    args = parse_args()
    for sub_set in ['train', 'valid']:
        convert_vis(args.input, args.output, sub_set)


if __name__ == '__main__':
    main()
