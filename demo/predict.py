import os
import os.path as osp
import sys
import torch
import numpy as np

sys.path.insert(0, osp.abspath('../'))

from tqdm import tqdm
import logging

from datasets.process.keypoints_ord import coco2posetrack_ord_infer
from tools.inference import inference_PE
from object_detector.YOLOv3.detector_yolov3 import inference_yolov3
from utils.utils_folder import list_immediate_childfile_paths
from utils.utils_video import video2images
from demo.dataset import MyDataset
from demo.mlp import MLP
from demo.save_pose import save_pose, load_pose

use_cached_model = "m-1667541321.6601925-epoch267-0.0004.pth"

zero_fill = 8
sample_max = 50
sample_inter = 5
n_epochs = 300
batch_size = 50
checkpoint_path = "./checkpoint"

logger = logging.getLogger(__name__)


def main():
    video()


def video():
    logger.info("Start")
    base_video_path = "./input"
    image_save_base = "./image"
    predict_video_path = osp.join(base_video_path,"predict")
    predict_video_list = list_immediate_childfile_paths(predict_video_path, ext=['mp4'])
    predict_image_save_dirs = []
    dic_images2video = {}
    # 1.Split the video into images
    print("1.Split the video into images")
    for video_path in tqdm(predict_video_list):
        video_name = osp.basename(video_path)
        temp = video_name.split(".")[0]
        image_save_path = os.path.join(image_save_base, temp)
        predict_image_save_dirs.append(image_save_path)
        dic_images2video[image_save_path.replace('\\', '/')] = video_path.replace('\\', '/')
        if osp.exists(image_save_path):
            continue
        video2images(video_path, image_save_path)  # jpg

    # 2. Person Instance detection
    print("2. Person Instance detection")
    logger.info("Person Instance detection in progress ...")
    video_candidates_test = {}
    for index, images_dir in enumerate(tqdm(predict_image_save_dirs)):
        video_name = osp.basename(images_dir)
        image_list = list_immediate_childfile_paths(images_dir, ext='jpg')
        video_candidates_list = []
        for image_path in image_list:
            candidate_bbox = inference_yolov3(image_path)
            for bbox in candidate_bbox:
                # bbox  - x, y, w, h
                video_candidates_list.append({"image_path": image_path,
                                              "bbox": bbox,
                                              "keypoints": None})
                break   #just take one bbox
        video_candidates_test[video_name] = {"candidates_list": video_candidates_list,
                                        "length": len(image_list)}                                  
    logger.info("Person Instance detection finish")


    # 3. Singe Person Pose Estimation for Training
    print("3. Singe Person Pose Estimation")
    logger.info("Single person pose estimation in progress ...")
    test_pose_estimatated_list = []
    name_list = []
    for video_name, video_info in tqdm(video_candidates_test.items()):
        pose_estimatated_seq = []
        video_candidates_list = video_info["candidates_list"]
        video_length = video_info["length"]
        for index, person_info in enumerate(video_candidates_list):
            if index >= sample_max:
                break
            if index % sample_inter != 0:
                continue
            image_path = person_info["image_path"]
            xywh_box = person_info["bbox"]
            image_idx = int(os.path.basename(image_path).replace(".jpg", ""))
            prev_idx, next_id = image_idx - 1, image_idx + 1
            if prev_idx < 0:
                prev_idx = 0
            if image_idx >= video_length - 1:
                next_id = video_length - 1
            prev_image_path = os.path.join(os.path.dirname(image_path), "{}.jpg".format(str(prev_idx).zfill(zero_fill)))
            next_image_path = os.path.join(os.path.dirname(image_path), "{}.jpg".format(str(next_id).zfill(zero_fill)))


            bbox = xywh_box
            keypoints = inference_PE(image_path, prev_image_path, next_image_path, bbox)
            # person_info["keypoints"] = keypoints.tolist()[0]

            #posetrack points
            mid_coord = coco2posetrack_ord_infer(keypoints[0])
            #coord change
            new_coord = changeCoord(mid_coord)
            
            pose_estimatated_seq.append(new_coord)
        if len(pose_estimatated_seq) != sample_max/sample_inter:
            temp = np.zeros(shape=(14,2)).tolist()
            for index in range(int(sample_max/sample_inter)-len(pose_estimatated_seq)):
                pose_estimatated_seq.append(temp)
        test_pose_estimatated_list.append(pose_estimatated_seq)
        name_list.append(video_name)
    predict_mlp(test_pose_estimatated_list, name_list)
    


def predict_mlp(test_pose_estimatated_list, name_list):
    test_pose_estimatated_list = torch.tensor(test_pose_estimatated_list, dtype=torch.float)
    model = MLP(checkpoint_path=checkpoint_path)
    model.load_checkpoint(subpath=use_cached_model)
    torch.no_grad()
    output = []
    for seq in test_pose_estimatated_list:
        temp = model(seq).argmax().item()
        output.append(temp)
    with open("output.txt",'w') as f:
        for index in range(len(name_list)):
            f.write(str(name_list[index])+"\t"+str(output[index])+"\n")
    print("output in file:"+"./output.txt")
        
        


def changeCoord(old_corrd):
    new_coord = []
    tempx = old_corrd[14][0]
    tempy = old_corrd[14][1]
    for item in old_corrd:
        new_coord.append([item[0]-tempx,item[1]-tempy])
    new_coord.pop()
    return new_coord


if __name__ == '__main__':
    main()
