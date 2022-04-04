import os
import random
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import cv2
import json
from PIL import Image
import torchvision.transforms as transforms

sample_random = random.Random()

class GTA_Events_Anomaly(Dataset):
    def __init__(self, data_root, logger, transform=None, train=False, patch=False, flip=False, scale=False):
        super(GTA_Events_Anomaly, self).__init__()

        self.transform = transform
        self.patch = patch
        self.flip = flip
        self.scale = scale
        self.train = train

        self.lstm_seq_frame_count = 8

        dataset_path = "/home/deepuser/deepnas/DISK2/DATASETS/GTA_Events_Dataset"
        if (train):
            txt_path = "/home/deepuser/deepnas/DISK2/DATASETS/GTA_Events_Dataset/train_videos.txt"
            logger.info("\nGTA-Events TRAIN Dataset Loading...")
            logger.info("Train Dataset txt path: " + txt_path)
        else:
            txt_path = "/home/deepuser/deepnas/DISK2/DATASETS/GTA_Events_Dataset/test_videos.txt"
            logger.info("\nGTA-Events TEST Dataset Loading...")
            logger.info("Test Dataset txt path: " + txt_path)
                
        txt_file = open(txt_path)
        _ = txt_file.readline()
        lines = txt_file.readlines()
        self.sequences = []
        for line in lines:
            video_name, anomaly_frame, anomaly_frame_amount = line.split(',')
            logger.info("      " + str(video_name) + " Loading...")
            anomaly_frame = int(anomaly_frame)
            anomaly_frame_amount = int(anomaly_frame_amount)
            seq_directory = os.path.join(dataset_path,video_name,video_name)

            frames = []
            for i in range (anomaly_frame-anomaly_frame_amount, anomaly_frame):
                image_name = str(i).zfill(10)
                frame_path = os.path.join(seq_directory, image_name + ".tiff")
                json_file_path = os.path.join(seq_directory, image_name + ".json")
                count=0
                coord_list=[]
                with open(json_file_path) as f:
                    json_file = json.load(f)
                    for j in json_file['Detections']:
                        if "IK_Head" not in j["Bones"]:
                            continue
                        count = count +1
                        x_coord = round(j["Bones"]["IK_Head"]["X"]*2560)
                        y_coord = round(j["Bones"]["IK_Head"]["Y"]*1440)
                        coord_list.append({'x':x_coord, 'y':y_coord})
                dict = {
                    "video_name": video_name,
                    "frame_path": frame_path,
                    "anomaly": False,
                    "person_coord_list": coord_list,
                    "person_count": count
                }
                frames.append(dict)

            for i in range (anomaly_frame, anomaly_frame+anomaly_frame_amount):
                image_name = str(i).zfill(10)
                frame_path = os.path.join(seq_directory, image_name + ".tiff")
                json_file_path = os.path.join(seq_directory, image_name + ".json")
                count=0
                coord_list=[]
                with open(json_file_path) as f:
                    json_file = json.load(f)
                    for j in json_file['Detections']:
                        if "IK_Head" not in j["Bones"]:
                            continue
                        count = count +1
                        x_coord = round(j["Bones"]["IK_Head"]["X"]*2560)
                        y_coord = round(j["Bones"]["IK_Head"]["Y"]*1440)
                        coord_list.append({'x':x_coord, 'y':y_coord})
                dict = {
                    "video_name": video_name,
                    "frame_path": frame_path,
                    "anomaly": True,
                    "person_coord_list": coord_list,
                    "person_count": count
                }
                frames.append(dict)

            for i in range(self.lstm_seq_frame_count, len(frames)):
                self.sequences.append(frames[i-self.lstm_seq_frame_count:i])   
        
        sample_random.shuffle(self.sequences)
        logger.info("Dataset Loaded!")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        seq_count = len(seq)

        img_list = []
        point_list = []
        anomaly_count=0
        for i in range(seq_count):
            frame = cv2.imread(seq[i]["frame_path"]) # 2560 x 1440 original size
            frame = cv2.resize(frame, (768,432), interpolation = cv2.INTER_AREA) #resize
            # frame = cv2.resize(frame, (1280,720), interpolation = cv2.INTER_AREA) #resize
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if self.transform is not None:
                img = self.transform(img)
            img = torch.Tensor(img)
            img_list.append(img)

            points = []
            for coords in seq[i]["person_coord_list"]:
                x = float(coords["x"])
                y = float(coords["y"])
                points.append([x, y])

            point = np.array(points)
            point = torch.Tensor(point)
            point_list.append(point)

        img = torch.stack(img_list)

        target = [{} for i in range(len(point_list))]
        for i, _ in enumerate(point_list):
            target[i]['point'] = torch.Tensor(point_list[i])
            target[i]['image_id'] = torch.Tensor([int(i)]).long()
            target[i]['labels'] = torch.ones([point_list[i].shape[0]]).long()
            if seq[i]["anomaly"] : anomaly_count = anomaly_count+1 

        if (anomaly_count > (seq_count/2)):
            anomaly_target = torch.tensor([1.0])
        else:
            anomaly_target = torch.tensor([0.0])

        return img, target, anomaly_target