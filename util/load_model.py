import pprint

def check_keys(model, pretrained_state_dict, logger, print_unuse=True):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = list(ckpt_keys - model_keys)
    missing_keys = list(model_keys - ckpt_keys)

    # remove num_batches_tracked
    for k in sorted(missing_keys):
        if 'num_batches_tracked' in k:
            missing_keys.remove(k)
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

def load_pretrain_net(model, pretrained_path, logger, print_unuse=True):
    logger.info('load pretrained whole network from {}'.format(pretrained_path))
    logger.info(pprint.pformat(model))

    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    pretrained_dict = pretrained_dict
    check_keys(model, pretrained_dict, logger, print_unuse=print_unuse)
    model.load_state_dict(pretrained_dict, strict=False)
    return model        