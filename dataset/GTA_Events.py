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

class GTA_Events(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        super(GTA_Events, self).__init__()

        self.transform = transform
        self.patch = patch
        self.flip = flip
        self.train = train

        self.lstm_seq_frame_count = 10

        dataset_path = "/home/deepuser/deepnas/DISK2/DATASETS/GTA_Events_Dataset"
        if (train):
            txt_path = "/home/deepuser/deepnas/DISK2/DATASETS/GTA_Events_Dataset/train_videos_test.txt"
        else:
            txt_path = "/home/deepuser/deepnas/DISK2/DATASETS/GTA_Events_Dataset/test_videos_test.txt"
                
        txt_file = open(txt_path)
        _ = txt_file.readline()
        lines = txt_file.readlines()
        self.sequences = []
        for line in lines:
            video_name, anomaly_frame, anomaly_frame_amount = line.split(',')
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
        
        sample_random.shuffle(self.self.sequences)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        seq = self.sequences[index]
        seq_count = len(seq)

        imgs_list = []
        point_targets_list = []
        anomaly_count=0
        for i in range(seq_count):
            frame = cv2.imread(seq[i]["frame_path"]) # 2560 x 1440 original size
            frame = cv2.resize(frame, (1280,720), interpolation = cv2.INTER_AREA) #resize
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            points = []
            for coords in seq[i]["person_coord_list"]:
                x = float(coords["x"])
                y = float(coords["y"])
                points.append([x, y])

            point = np.array(points)

            if self.transform is not None:
                img = self.transform(img)

            if self.train:
                # data augmentation -> random scale
                scale_range = [0.7, 1.3]
                min_size = min(img.shape[1:])
                scale = random.uniform(*scale_range)
                # scale the image and points
                if scale * min_size > 128:
                    img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                    point *= scale
            # random crop augumentaiton
            if self.train and self.patch:
                img, point = random_crop(img, point)
                for i, _ in enumerate(point):
                    point[i] = torch.Tensor(point[i])
            # random flipping
            if random.random() > 0.5 and self.train and self.flip:
                # random flip
                img = torch.Tensor(img[:, :, :, ::-1].copy())
                for i, _ in enumerate(point):
                    point[i][:, 0] = 128 - point[i][:, 0]

            point = [point]

            img = torch.Tensor(img)
            # pack up related infos
            target = [{} for i in range(len(point))]
            for i, _ in enumerate(point):
                target[i]['point'] = torch.Tensor(point[i])
                image_id = int(i)
                image_id = torch.Tensor([image_id]).long()
                target[i]['image_id'] = image_id
                target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

            if seq[i]["anomaly"] : anomaly_count = anomaly_count+1 

            imgs_list.append(img)
            point_targets_list.append(target)

        if (anomaly_count > (seq_count/2)):
            anomaly_target = torch.tensor([1.0])
        else:
            anomaly_target = torch.tensor([0.0])

        imgs = torch.stack(imgs_list)
        point_targets = torch.stack(point_targets_list)
        return imgs, point_targets, anomaly_target

def random_crop(img, den, num_patch=4):
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        # copy the cropped points
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        # shift the corrdinates
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h

        result_den.append(record_den)

    return result_img, result_den