# removed Copyright for double blind review TODO

import argparse
import copy
import os
import os.path as osp
from collections import defaultdict

import mmcv
from tqdm import tqdm
import os
import numpy as np
import PIL
import cv2
import shutil


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
    #new annotation file
    VIS = defaultdict(list)

    root = '../data/sailvos/' #TODO set path to SAILVOS here

    #here load the json file
    official_anns = mmcv.load(osp.join(ann_dir, f'{mode}.json')) #has to be train.json or valid.json

    print("before categories")
    VIS['categories'] = copy.deepcopy(official_anns['categories'])  #categories do not need to be changed
    print("before models")
    VIS['models'] = copy.deepcopy(official_anns['models'])  # models do not need to be changed

    #process all frames to get the frame cuts in every video
    cut_dic = defaultdict(list)# for every video save all the cut frames
    global_video_id = 0
    for video_info in official_anns['videos']:
        cut_list = [0]#initialze the cut list of one video # 1 and not 0 so i do not  have to do condition and cn always use -1
        num_frames = video_info['num_frames']
        video_id = video_info['id'] #integer
        video_name = video_info['name']
        #print("The type is : ", type(video_id))
        print(video_info)
        imgs_name_in_a_cut_video = []
        imgs_ids_in_a_cut_video = []
        #print("new video cut list",cut_list )
        for frame_num in range(num_frames):
            print("current frame", frame_num, "from ",num_frames )
            for image_info in official_anns['images']:
                first_postion = image_info['file_name'].find('/')  # find the first "/" in filename
                old_video_name = image_info['file_name'][0:first_postion]
                if video_name==old_video_name and image_info['frame_id'] == frame_num: #image_info['video_id'] == video_id the video is looks like it is not unique
                    #store the images names without the video name for a cut video # for example /images/000003.bmp

                    img_string = image_info['file_name'][first_postion:] #
                    imgs_name_in_a_cut_video.append(img_string)#image_info['id']
                    imgs_ids_in_a_cut_video.append(image_info['id'])
                    #print("imgs_name_in_a_cut_video",imgs_name_in_a_cut_video)
                    #print("imgs_ids_in_a_cut_video ",imgs_ids_in_a_cut_video)

                    if frame_num==0:
                        #first frame shoud not be a cut frame
                        if os.path.exists(root+image_info['file_name']): #+mode+'/RGB_frames/'
                            print("first frame path exists")
                            last_frame=PIL.Image.open(root+image_info['file_name']) #+mode+'/RGB_frames/'
                            last_frame_np= np.array(last_frame)
                            last_frame_hsv =  cv2.split(cv2.cvtColor(last_frame_np, cv2.COLOR_BGR2HSV))
                    elif frame_num==(num_frames-1): # because range starts from 0
                        print("reached the last frame of the video") # the last frame should always be a cut frame
                        #print("cut_list",cut_list)
                        new_video_name = old_video_name + "__" + str(frame_num)
                        new_number_frame_in_video = frame_num - cut_list[-1] +1  #(+1 to include the last frame) # only -1 because there is no append before
                        global_video_id = global_video_id + 1  # the video_name will be the unique identifier in the convert process
                        #add to dictionary
                        video = dict(
                            id=global_video_id,
                            name=new_video_name,  # here changing vid_name to name
                            width=image_info['width'],  # added
                            height=image_info['height'],  # added
                            num_frames=new_number_frame_in_video,  # kept
                            #num_objs=video_info['num_objs'] # not needed
                            )
                        print(video)
                        VIS['videos'].append(video)
                        # label all the images in this video
                        print("imgs_name_in_a_cut_video", imgs_name_in_a_cut_video)
                        print("imgs_ids_in_a_cut_video ",imgs_ids_in_a_cut_video)
                        print("len(imgs_name_in_a_cut_video)",len(imgs_name_in_a_cut_video) ,"new_number_frame_in_video",new_number_frame_in_video )
                        for frame_counter in range(new_number_frame_in_video):  # frames count start from 0
                            file_name = new_video_name + imgs_name_in_a_cut_video[frame_counter]
                            image = dict(
                                id=imgs_ids_in_a_cut_video[frame_counter],  # should not chnage!
                                video_id=global_video_id,
                                file_name=file_name,
                                frame_id=frame_counter,
                                width=image_info['width'],
                                height=image_info['height'],
                            )
                            print(image)
                            VIS['images'].append(image)
                            # move the image in the new folder
                            source =root + old_video_name + imgs_name_in_a_cut_video[frame_counter]
                            destination = root + file_name

                            makevideofolder = root  + new_video_name
                            makeimagesfolder = root  + new_video_name + '/'+ 'images'
                            if not os.path.exists(makevideofolder):
                                os.mkdir(makevideofolder)
                            if not os.path.exists(makeimagesfolder):
                                os.mkdir(makeimagesfolder)
                            #print(os.path.exists(source))
                            if os.path.exists(source):
                                # i did not want to print something
                                #print("source path exists and i can move the image")
                                shutil.move(source, destination)
                            else:
                                print("source path DOES NOT EXIST and i can not move the image")
                        imgs_name_in_a_cut_video = []
                        imgs_ids_in_a_cut_video = []
                        #print("imgs_name_in_a_cut_video", imgs_name_in_a_cut_video)
                        #print("imgs_ids_in_a_cut_video ", imgs_ids_in_a_cut_video)
                    else:
                        if os.path.exists(root + image_info['file_name']):
                            #print("current frame path exists")
                            curr_frame = PIL.Image.open(root + image_info['file_name'])
                            curr_frame_np = np.array(curr_frame)
                            current_frame_hsv = cv2.split(cv2.cvtColor(curr_frame_np, cv2.COLOR_BGR2HSV))
                        else:
                            print("current frame path DOES NOT EXIST")
                        #score calculation like in PySceneDetect content_detector
                        current_frame_hsv = [x.astype(np.int32) for x in current_frame_hsv]
                        last_frame_hsv = [x.astype(np.int32) for x in last_frame_hsv]
                        delta_hsv = [0, 0, 0, 0]
                        for i in range(3):
                            num_pixels = current_frame_hsv[i].shape[0] * current_frame_hsv[i].shape[1]
                            delta_hsv[i] = np.sum(
                                np.abs(current_frame_hsv[i] - last_frame_hsv[i])) / float(num_pixels)
                        delta_hsv[3] = sum(delta_hsv[0:3]) / 3.0
                        frame_score = delta_hsv[3]
                        last_frame_hsv = current_frame_hsv
                        if  frame_score >= 27.0 and ((frame_num - cut_list[len(cut_list)-1]) >= 5) and ((num_frames-1-frame_num)>5 ):
                            #print("frame_score", frame_score)
                            cut_list.append(frame_num)
                            #print(len(cut_list))
                            #new cut new video
                            #print("cut_list",cut_list)
                            new_video_name = old_video_name + "__" + str(frame_num)
                            new_number_frame_in_video = frame_num - cut_list[len(cut_list)-2]    # current cut frame- last cut frame
                            global_video_id = global_video_id + 1  # the video_name will be the unique identifier in the convert process
                            # add to dictionary
                            video = dict(
                                id=global_video_id,
                                name=new_video_name,  # here changing vid_name to name
                                width=image_info['width'],  # added
                                height=image_info['height'],  # added
                                num_frames=new_number_frame_in_video,  # kept
                                # num_objs=video_info['num_objs'] # not needed
                            )
                            print(video)
                            VIS['videos'].append(video)

                            print("imgs_name_in_a_cut_video", imgs_name_in_a_cut_video)
                            print("imgs_ids_in_a_cut_video ", imgs_ids_in_a_cut_video)
                            print(
                            "len(imgs_name_in_a_cut_video)", len(imgs_name_in_a_cut_video), "new_number_frame_in_video",
                            new_number_frame_in_video)
                            # label all the images in this video
                            for frame_counter in range(new_number_frame_in_video):  # frames count start from 0
                                file_name = new_video_name + imgs_name_in_a_cut_video[frame_counter]
                                image = dict(
                                    id=imgs_ids_in_a_cut_video[frame_counter],  # should not chnage!
                                    video_id=global_video_id,
                                    file_name=file_name,
                                    frame_id=frame_counter,
                                    width=image_info['width'],
                                    height=image_info['height'],
                                )
                                print(image)
                                VIS['images'].append(image)

                                # move the image in the new folder
                                source = root  + old_video_name + imgs_name_in_a_cut_video[frame_counter]
                                destination = root  + file_name

                                makevideofolder = root  + new_video_name
                                makeimagesfolder = root + new_video_name + '/' + 'images'
                                if not os.path.exists(makevideofolder):
                                    os.mkdir(makevideofolder)
                                if not os.path.exists(makeimagesfolder):
                                    os.mkdir(makeimagesfolder)
                                #print(os.path.exists(source))
                                if os.path.exists(source):
                                    #print("source path exists and i can move the image")
                                    shutil.move(source, destination)
                                else:
                                    print("source path DOES NOT EXISTS and i can not move the image")
                            imgs_name_in_a_cut_video = [imgs_name_in_a_cut_video[-1]]
                            imgs_ids_in_a_cut_video = [imgs_ids_in_a_cut_video[-1]]
                            #print("imgs_name_in_a_cut_video", imgs_name_in_a_cut_video) #because the cut frame belongs to the next video
                            #print("imgs_ids_in_a_cut_video ", imgs_ids_in_a_cut_video)

        #print("cut_list",cut_list) # []
        cut_dic[video_id].append(cut_list)
        print("cut_dic",cut_dic) #{9: [[15,30,  ]]}
        #print("cut_dic",cut_dic[9])# [[15,30,  ]]
        #print("cut_dic",cut_dic[9][0]) # [15,30, ]
        #print("cut_dic",cut_dic[9][0][0]) # 15
        #print(VIS['videos'])
        #print(VIS['images'])
        #here just for checking but move outside the loop

        #just_two_videos_to_see_img_not_resetting = just_two_videos_to_see_img_not_resetting +1
        #if just_two_videos_to_see_img_not_resetting ==40:
        #    print(stop)

    print("cut_dic", cut_dic)

    #now categories, models , videos, images are converted prepare annotations
    imgToVideo = defaultdict(list)
    for local_img in VIS['images']:
        imgToVideo[local_img['id']] = local_img['video_id'] #'FROM THE IMAGE ID I CAN GET THE VIDEO ID'

    records = dict(global_instance_id=1)
    old_frames = 0
    new_frames = 0
    for vids in VIS['videos']:
        vid_id = vids['id']
        object_ids_in_video = dict()  # when obj_id gets replaced with an instance id map the two values
        #print("video id is ", vid_id)
        new_frames = vids['num_frames'] + new_frames
        #print("old_frames are ", old_frames)
        #print("new_frames are ", new_frames)
        for imageid in range(old_frames - 5, new_frames + 5):  # more than 26872 for valid#
            # changing the annotations: instance_id gets obj_id and add video key
            for anns in official_anns['annotations']:
                if anns['image_id'] == imageid and imgToVideo[anns['image_id']] == vid_id:  # anns['image_id'] == frame_id and i do not  have a frame id
                    #print("imageid id is ", imageid)
                    #print("vid_id in loop is ", vid_id)
                    object_id = anns['instance_id']
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
