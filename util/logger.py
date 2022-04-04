import logging
import time
from pathlib import Path
import math
import pprint

def create_logger(args, phase='train'):
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print('=> creating {}'.format(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = 'CMP_{}_{}.log'.format(time_str, phase)
    final_log_file = output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger, time_str

def print_speed(i, i_time, n, logger):
    """print_speed(index, index_time, total_iteration)"""
    average_time = i_time
    remaining_time = (n - i) * average_time
    remaining_day = math.floor(remaining_time / 86400)
    remaining_hour = math.floor(remaining_time / 3600 - remaining_day * 24)
    remaining_min = math.floor(remaining_time / 60 - remaining_day * 1440 - remaining_hour * 60)
    logger.info('\nProgress: %d / %d [%d%%], Speed: %.3f s/iter, ETA %d:%02d:%02d (D:H:M)' % (i, n, 
                 i/n*100, average_time, remaining_day, remaining_hour, remaining_min))
    logger.info('PROGRESS: {:.2f}%\n'.format(100 * i / n))

def check_trainable(model, logger):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info('trainable params:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)
    assert len(trainable_params) > 0, 'no trainable parameters'
    return trainable_params

def check_keys(model, pretrained_state_dict, logger, print_unuse=True):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = list(ckpt_keys - model_keys)
    missing_keys = list(model_keys - ckpt_keys)

    # # remove num_batches_tracked
    # for k in sorted(missing_keys):
    #     if 'num_batches_tracked' in k:
    #         missing_keys.remove(k)
    logger.info("<<<<<<<<<<<<<<<<<< ------------------------- >>>>>>>>>>>>>>>>>>>>>>>")
    logger.info(pprint.pformat('pretrained_dict keys:{}'.format(ckpt_keys)))
    logger.info("<<<<<<<<<<<<<<<<<< ------------------------- >>>>>>>>>>>>>>>>>>>>>>>")
    logger.info(pprint.pformat('missing keys:{}'.format(missing_keys)))
    if print_unuse:
        logger.info("<<<<<<<<<<<<<<<<<< ------------------------- >>>>>>>>>>>>>>>>>>>>>>>")
        logger.info(pprint.pformat('unused checkpoint keys:{}'.format(unused_pretrained_keys)))
    logger.info("<<<<<<<<<<<<<<<<<< ------------------------- >>>>>>>>>>>>>>>>>>>>>>>")
    logger.info(pprint.pformat('used keys:{}'.format(used_pretrained_keys)))

    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True