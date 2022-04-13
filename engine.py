# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import math
import os
import sys
from typing import Iterable
from util.logger import print_speed

import torch

import util.misc as utils
from util.misc import NestedTensor
import numpy as np
import time
import torchvision.transforms as standard_transforms
import cv2

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def vis(samples, targets, pred, vis_dir, des=None):
    '''
    samples -> tensor: [batch, 3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    '''
    gts = [t['point'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    # draw one by one
    for idx in range(samples.shape[0]):
        sample = restore_transform(samples[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        max_len = np.max(sample_gt.shape)

        size = 2
        # draw gt
        for t in gts[idx]:
            sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
        # draw predictions
        for p in pred[idx]:
            sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        name = targets[idx]['image_id']
        # save the visualized images
        if des is not None:
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_gt.jpg'.format(int(name), 
                                                des, len(gts[idx]), len(pred[idx]))), sample_gt)
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_pred.jpg'.format(int(name), 
                                                des, len(gts[idx]), len(pred[idx]))), sample_pred)
        else:
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_gt.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_gt)
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_pred.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_pred)

# the training routine
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # iterate all training samples
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # forward
        outputs = model(samples)
        # calc the losses
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce all losses
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        # backward
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # update logger
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_batch(model: torch.nn.Module, criterion: torch.nn.Module,
                    logger, tensorboard_writer_dict, log_print_freq,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, end_epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()

    epoch_total_loss = 0
    epoch_total_point_loss = 0
    epoch_total_anomaly_loss = 0
    t2 = time.time()    
    for iter, data in enumerate(data_loader):
        t1 = time.time()
        data_time = t1 - t2
        imgs, point_targets, anomaly_target = data
        samples = imgs.squeeze(0).to(device)
        anomaly_target = anomaly_target.squeeze(0).type(torch.FloatTensor).to(device)
        targets = [{k: v.squeeze(0).to(device) for k, v in t.items()} for t in point_targets]

        outputs = model(samples)

        point_losses, anomaly_loss = criterion(outputs, targets, anomaly_target)
        weight_dict = criterion.weight_dict
        point_loss = sum(point_losses[k] * weight_dict[k] for k in point_losses.keys() if k in weight_dict)
        point_loss = torch.mean(point_loss)
        losses = point_loss + anomaly_loss
        #losses = anomaly_loss

        epoch_total_loss += losses.item()
        epoch_total_point_loss += point_loss.item()
        epoch_total_anomaly_loss += anomaly_loss.item()

        # backward
        optimizer.zero_grad()
        losses.backward()
        # if max_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        t2 = time.time()
        process_time = t2 - t1

        if ((iter + 1) % log_print_freq == 0):
            logger.info('\nTRAIN - Epoch: [{0}][{1}/{2}] lr: {lr:.7f}  Process Time: {process_time:.3f}s Data Time:{data_time:.3f}s  ' \
              'Point Total Loss: {point_total_loss:.5f}  Point Regression Loss: {point_reg_loss:.5f}  Point Cls Loss: {point_cls_loss:.5f}  ' \
              'Anomaly Loss: {anomaly_loss:.5f}  Total Loss: {total_loss:.5f}'.format(epoch, iter+1, len(data_loader), lr=optimizer.param_groups[0]["lr"], 
              process_time=process_time, data_time=data_time, point_total_loss=point_loss.item(), point_reg_loss=point_losses["loss_point"].item(), 
              point_cls_loss=point_losses["loss_ce"].item(), anomaly_loss=anomaly_loss.item(), total_loss=losses.item()))

            print_speed((epoch) * len(data_loader) + iter + 1, process_time + data_time,
                        end_epoch * len(data_loader), logger)

        writer = tensorboard_writer_dict['writer']
        global_steps = tensorboard_writer_dict['train_global_steps']
        writer.add_scalars('Train_Losses', {'total_loss': losses.item(), 'point_total_loss': point_loss.item(), 'point_cls_loss': point_losses["loss_ce"].item(),
                            'point_reg_loss': point_losses["loss_point"].item(), 'anomaly_loss': anomaly_loss},global_steps)  
        writer.add_scalars('Progress_Info', {'epoch' : epoch, 
                            'iteration' : iter + 1},global_steps) 
        writer.add_scalars('Time_Info', {'data_time' : data_time, 
                            'process_time' : process_time},global_steps)                                   
        tensorboard_writer_dict['train_global_steps'] = global_steps + 1

    epoch_average_loss = epoch_total_loss/iter
    epoch_average_point_loss = epoch_total_point_loss/iter
    epoch_average_anomaly_loss = epoch_total_anomaly_loss/iter
    return epoch_average_point_loss, epoch_average_anomaly_loss, epoch_average_loss


@torch.no_grad()
def evaluate_crowd_no_overlap(model, data_loader, device, vis_dir=None):
    model.eval()
    
    threshold = 0.5

    count_maes = []
    count_mses = []
    for samples, targets in data_loader:
        samples = samples.to(device)
        outputs = model(samples)

        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        predict_cnt = int((outputs_scores > threshold).sum())
        gt_cnt = targets[0]['point'].shape[0]
        
        if vis_dir is not None: 
            outputs_points = outputs['pred_points'][0]
            points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
            vis(samples, targets, [points], vis_dir)

        # accumulate MAE, MSE
        count_mae = abs(predict_cnt - gt_cnt)
        count_mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        count_maes.append(float(count_mae))
        count_mses.append(float(count_mse))
    # calc MAE, MSE
    count_mae = np.mean(count_maes)
    count_mse = np.sqrt(np.mean(count_mses))

    return count_mae, count_mse


@torch.no_grad()
def evaluate_crowd_no_overlap_batch(model, data_loader, device, vis_dir=None):
    model.train()
    threshold = 0.5

    count_maes = []
    count_mses = []
    anomaly_accuracy = []
    for imgs, point_targets, anomaly_target in data_loader:
        samples = imgs.squeeze(0).to(device)
        anomaly_target = anomaly_target.squeeze(0).type(torch.FloatTensor)
        targets = [{k: v.squeeze(0).to(device) for k, v in t.items()} for t in point_targets]
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        predict_cnt = int((outputs_scores > threshold).sum())
        gt_cnt = targets[0]['point'].shape[0]

        if vis_dir is not None: 
            outputs_points = outputs['pred_points'][0]
            points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
            vis(samples, targets, [points], vis_dir)

        count_mae = abs(predict_cnt - gt_cnt)
        count_mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        count_maes.append(float(count_mae))
        count_mses.append(float(count_mse))

        anomaly_score = outputs['anomaly']
        anomaly_score_binary = 1.0 if (anomaly_score > 0.5) else 0.0
        accuracy = 1.0 if anomaly_score_binary == anomaly_target else 0.0
        anomaly_accuracy.append(accuracy)

    # calc MAE, MSE
    count_mae = np.mean(count_maes)
    count_mse = np.sqrt(np.mean(count_mses))
    anomaly_accuracy = np.mean(anomaly_accuracy)

    return count_mae, count_mse, anomaly_accuracy