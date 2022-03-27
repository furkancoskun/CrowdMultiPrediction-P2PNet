import logging
import time
from pathlib import Path
import math

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
    logger.info('Progress: %d / %d [%d%%], Speed: %.3f s/iter, ETA %d:%02d:%02d (D:H:M)\n' % (i, n, 
                 i/n*100, average_time, remaining_day, remaining_hour, remaining_min))
    logger.info('\nPROGRESS: {:.2f}%\n'.format(100 * i / n))