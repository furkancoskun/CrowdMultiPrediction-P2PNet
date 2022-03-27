import argparse
import datetime
import random
import time
from pathlib import Path
import os
from models.p2pnet_withAnomaly import P2PNetWithAnomaly
import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.logger import create_logger
import pprint
from dataset import build_dataset
from engine import *
from models import build_model
import os
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training P2PNet', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_fpn', default=1e-5, type=float)
    parser.add_argument('--lr_regression_classification', default=1e-5, type=float)
    parser.add_argument('--lr_lstm_encoder', default=1e-4, type=float)
    parser.add_argument('--lr_anomaly_cls_head', default=1e-4, type=float)    

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=3500, type=int)
    parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--model_type', type=str, default="P2PNetWithAnomaly",
                        help="P2PNetWithAnomaly or P2PNet")

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)

    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--row', default=3, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=3, type=int,
                        help="line number of anchor points")

    # dataset parameters
    parser.add_argument('--dataset_file', default='SHHA')
    parser.add_argument('--data_root', default='./new_public_density_data',
                        help='path where the dataset is')
    
    parser.add_argument('--output_dir', default='./log',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./ckpt',
                        help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='./runs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_print_freq', default=10, type=int,
                        help='frequency of log printing')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')

    return parser

# python train.py --data_root onemsiz --dataset_file GTA_EVENTS --epochs 3500 --lr_drop 3500 --output_dir ./logs --checkpoints_dir ./weights 
# --tensorboard_dir ./logs --lr 0.0005 --lr_backbone 0.0005 --batch_size 4 --eval_freq 10 --gpu_id 3

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    os.chdir("/home/deepuser/deepnas/DISK2/furkan_workspace/CrowdMultiPrediction-P2PNet")

    logger, time_str = create_logger(args, phase='train')

    tensorboard_writer_path = os.path.join(args.tensorboard_dir, "train_" + time_str)
    writer_dict = {
        'writer': SummaryWriter(log_dir=tensorboard_writer_path),
        'train_global_steps': 0,
        'validation_global_steps': 0,
    }
    logger.info(pprint.pformat('args:{}'.format(args)))

    device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # get the P2PNet model
    model, criterion = build_model(args, training=True)
    # move to GPU
    model.to(device)
    criterion.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:', n_parameters)
    # use different optimation params for different parts of the model
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model.named_parameters() if "fpn" in n and p.requires_grad],
            "lr": args.lr_fpn,
        },
        {
            "params": [p for n, p in model.named_parameters() if ("regression" or "classification") in n and p.requires_grad],
            "lr": args.lr_regression_classification,
        },
        {
            "params": [p for n, p in model.named_parameters() if "lstm_encoder" in n and p.requires_grad],
            "lr": args.lr_lstm_encoder,
        },    
        {
            "params": [p for n, p in model.named_parameters() if "anomalyClsHead" in n and p.requires_grad],
            "lr": args.lr_anomaly_cls_head,
        },            
        {   "params": [p for n, p in model.named_parameters() if ("backbone" or "fpn" or "regression" or "classification" or
                      "lstm_encoder" or "anomalyClsHead")  not in n and p.requires_grad],
            "lr": args.lr,
        },
    ]
    # Adam is used by default
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # create the dataset
    loading_data = build_dataset(args=args)
    # create the training and valiation set
    train_set, val_set = loading_data(args.data_root)
    # create the sampler used during training
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    # the dataloader for training
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    # resume the weights and training state if exists
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    logger.info("Start training")
    start_time = time.time()
    # save the performance during the training
    mae = []
    mse = []
    
    step = 0
    # training starts here
    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()
        stat = train_one_epoch(
            model, criterion, logger, writer_dict, args.log_print_freq, data_loader_train, 
            optimizer, device, epoch, args.epochs, args.clip_max_norm)

        logger.info("EPOCH loss/loss@{}: {}".format(epoch, stat['loss']))
        logger.info("EPOCH loss/loss_ce@{}: {}".format(epoch, stat['loss_ce']))

        t2 = time.time()
        logger.info('[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        
        # change lr according to the scheduler
        lr_scheduler.step()
        # save latest weights every epoch
        checkpoint_latest_path = os.path.join(args.checkpoints_dir, 'latest.pth')
        torch.save({
            'model': model.state_dict(),
        }, checkpoint_latest_path)

        # run evaluation
        if epoch % args.eval_freq == 0 and epoch != 0:
            t1 = time.time()
            result = evaluate_crowd_no_overlap(model, data_loader_val, device)
            t2 = time.time()

            mae.append(result[0])
            mse.append(result[1])

            logger.info('=======================================test=======================================')
            logger.info("mae:", result[0], "mse:", result[1], "time:", t2 - t1, "best mae:", np.min(mae), )
            logger.info("mae:{}, mse:{}, time:{}, best mae:{}".format(result[0], 
                                result[1], t2 - t1, np.min(mae)))
            logger.info('=======================================test=======================================')

            writer.add_scalar('metric/mae', result[0], step)
            writer.add_scalar('metric/mse', result[1], step)
            step += 1

            # save the best model since begining
            if abs(np.min(mae) - result[0]) < 0.01:
                checkpoint_best_path = os.path.join(args.checkpoints_dir, 'best_mae.pth')
                torch.save({
                    'model': model.state_dict(),
                }, checkpoint_best_path)

    # total time for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    writer_dict['writer'].close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)