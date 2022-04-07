import argparse
import random
import time
import os
import torch
from torch.utils.data import DataLoader

from util.logger import *
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
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--lr_fpn', default=1e-4, type=float)
    parser.add_argument('--lr_regression_classification', default=1e-4, type=float)
    parser.add_argument('--lr_lstm_encoder', default=5e-4, type=float)
    parser.add_argument('--lr_anomaly_cls_head', default=5e-4, type=float)    

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr_drop', default=50, type=int)
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
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    # dataset parameters
    parser.add_argument('--dataset_file', default='GTA_EVENTS_ANOMALY')
    parser.add_argument('--data_root', default='GTA_EVENTS_ANOMALY',
                        help='path where the dataset is')
    
    parser.add_argument('--output_dir', default='./logs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./weights',
                        help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='./logs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_print_freq', default=10, type=int,
                        help='frequency of log printing')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--initialize_weights', default='', help='initialize_weight from a weight(.pth) file')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--gpu_id', default=2, type=int, help='the gpu used for training')

    return parser

# python train_anomaly.py --epochs 3500 --lr_drop 3500 --output_dir ./logs --checkpoints_dir ./weights 
# --tensorboard_dir ./logs --batch_size 1 --initialize_weights weights/SHTechA.pth --eval_freq 1 --gpu_id 1
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    os.chdir("/home/deepuser/deepnas/DISK2/furkan_workspace/CrowdMultiPrediction-P2PNet")

    logger, time_str = create_logger(args, phase='anomaly_train')

    tensorboard_writer_path = os.path.join(args.tensorboard_dir, "anomaly_train_" + time_str)
    writer_dict = {
        'writer': SummaryWriter(log_dir=tensorboard_writer_path),
        'train_global_steps': 0,
        'test_global_steps': 0,
    }
    logger.info(pprint.pformat('args:{}'.format(args)))

    device = torch.device('cuda')

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args, training=True)
    # move to GPU
    model.to(device)
    criterion.to(device)

    logger.info('\n==========check trainable params==========')
    check_trainable(model, logger)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('\nnumber of params:{}'.format(n_parameters))

    # use different optimation params for different parts of the model
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone
        },
        {
            "params": [p for n, p in model.named_parameters() if "fpn" in n and p.requires_grad],
            "lr": args.lr_fpn
        },
        {
            "params": [p for n, p in model.named_parameters() if ("regression" or "classification") in n and p.requires_grad],
            "lr": args.lr_regression_classification
        },
        {
            "params": [p for n, p in model.named_parameters() if "lstm_encoder" in n and p.requires_grad],
            "lr": args.lr_lstm_encoder
        },    
        {
            "params": [p for n, p in model.named_parameters() if "anomalyClsHead" in n and p.requires_grad],
            "lr": args.lr_anomaly_cls_head
        },
    ]

    optimizer = torch.optim.Adam(param_dicts, lr=args.lr)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.8)

    loading_data = build_dataset(args=args)
    train_set, val_set = loading_data(args.data_root, logger)
    data_loader_train = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                   pin_memory=True, sampler=None, drop_last=True)
    data_loader_val = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                   pin_memory=True, sampler=None, drop_last=True)

    if args.resume:
        logger.info('\n=============Loading from resume=============')
        logger.info('resume path: ' + str(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        check_keys(model, checkpoint['model'], logger, print_unuse=True)
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.initialize_weights:
        logger.info('\n=============Loading from initial weights=============')
        logger.info('initial weights path: ' + str(args.initialize_weights))
        checkpoint = torch.load(args.initialize_weights, map_location='cpu')
        check_keys(model, checkpoint['model'], logger, print_unuse=True)
        model.load_state_dict(checkpoint['model'], strict=False)

    logger.info('\n=============Start Training=============')
    mae = []
    mse = []
    anomaly_accuracy = []

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch_batch( model, criterion, logger, writer_dict, args.log_print_freq, data_loader_train, 
            optimizer, device, epoch, args.epochs, args.clip_max_norm)
        
        
        lr_scheduler.step()
        
        checkpoint_latest_path = os.path.join(args.checkpoints_dir, 'anomaly_model_e%d.pth' % (epoch + 1))
        torch.save({
            'model': model.state_dict(),
        }, checkpoint_latest_path)

        # run evaluation
        if epoch % args.eval_freq == 0:
            t1 = time.time()
            result = evaluate_crowd_no_overlap_batch(model, data_loader_val, device)
            t2 = time.time()

            mae.append(result[0])
            mse.append(result[1])
            anomaly_accuracy.append(result[2])

            logger.info('===================test===================')
            logger.info('count_mae:{count_mae:.5f}, count_mse:{count_mse:.5f}, anomaly_accuracy:{anomaly_accuracy:.5f}, best_mae:{best_mae:.5f}, ' \
                        'best_anomaly_accuracy:{best_anomaly_accuracy:.5f}, time:{time:.5f}s'.format(count_mae=result[0], count_mse=result[1],
                        anomaly_accuracy=result[2], time=(t2-t1), best_mae=np.min(mae), best_anomaly_accuracy=np.max(anomaly_accuracy)))
            logger.info('===================test===================')

            writer = writer_dict['writer']
            global_test_steps = writer_dict['test_global_steps']
            writer.add_scalars('Test_Metrics', {'count_mae' : result[0], 'count_mse' : result[1],
                                                'anomaly_accuracy' : result[2],},global_test_steps)
            writer_dict['test_global_steps'] = global_test_steps + 1

            if abs(np.min(mae) - result[0]) < 0.01:
                checkpoint_best_path = os.path.join(args.checkpoints_dir, 'anomaly_model_best_mae.pth')
                torch.save({
                    'model': model.state_dict(),
                }, checkpoint_best_path)

            if abs(np.min(anomaly_accuracy) - result[2]) < 0.01:
                checkpoint_best_path = os.path.join(args.checkpoints_dir, 'anomaly_model_best_anomaly_accuracy.pth')
                torch.save({
                    'model': model.state_dict(),
                }, checkpoint_best_path)

    writer_dict['writer'].close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)