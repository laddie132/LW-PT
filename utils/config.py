#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import os
import sys
import yaml
import json
import torch.backends.cudnn
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter
import logging
import logging.config
from utils.functions import set_seed

logger = logging.getLogger(__name__)


def init_env(config_path, in_infix, out_infix, writer_suffix, gpuid=None):
    config_path = os.path.join('config', config_path)
    logger.info('loading config file: {}'.format(config_path))
    logger.info('in_infix: {}'.format(in_infix))
    logger.info('out_infix: {}'.format(out_infix))
    game_config = read_config(config_path, in_infix, out_infix)

    # config in logs
    logger.debug(json.dumps(game_config, indent=2))

    # set multi-processing: bugs in `list(dataloader)`
    # see more on `https://github.com/pytorch/pytorch/issues/973`
    torch.multiprocessing.set_sharing_strategy('file_system')

    # set random seed
    set_seed(game_config['global']['random_seed'])

    # gpu
    enable_cuda = torch.cuda.is_available() and gpuid is not None
    device = torch.device("cuda" if enable_cuda else "cpu")
    if enable_cuda:
        torch.cuda.set_device(gpuid)
        torch.backends.cudnn.deterministic = True
    logger.info("CUDA #{} is avaliable".format(gpuid)
                if enable_cuda else "CUDA isn't avaliable")

    # summary writer
    writer = SummaryWriter(log_dir=game_config['checkpoint'][writer_suffix])

    return game_config, enable_cuda, device, writer


def init_logging(config_path='config/logging.yaml', out_infix='default'):
    """
    initial logging module with config
    :param out_infix:
    :param config_path:
    :return:
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.Loader)

        out_prefix = 'outputs/' + out_infix + '/'
        if not os.path.exists(out_prefix):
            os.makedirs(out_prefix)

        config['handlers']['info_file_handler']['filename'] = out_prefix + 'debug.log'
        config['handlers']['time_file_handler']['filename'] = out_prefix + 'debug.log'
        config['handlers']['error_file_handler']['filename'] = out_prefix + 'error.log'

        logging.config.dictConfig(config)
    except IOError:
        sys.stderr.write('logging config file "%s" not found' % config_path)
        logging.basicConfig(level=logging.DEBUG)


def read_config(config_path='config/config.yaml', in_infix='default', out_infix='default'):
    """
    store the global parameters in the project
    :param in_infix:
    :param out_infix:
    :param config_path:
    :return:
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.Loader)

        out_prefix = 'outputs/' + out_infix + '/'
        if not os.path.exists(out_prefix):
            os.makedirs(out_prefix)

        in_prefix = 'outputs/' + in_infix + '/'
        assert os.path.exists(in_prefix)

        checkpoint = {'qt_log_path': out_prefix + 'qt_logs',
                      'in_qt_weight_path': in_prefix + 'qt_weight.pt',
                      'in_qt_checkpoint_path': in_prefix + 'qt_checkpoint',
                      'out_qt_weight_path': out_prefix + 'qt_weight.pt',
                      'out_qt_checkpoint_path': out_prefix + 'qt_checkpoint',
                      'cls_log_path': out_prefix + 'cls_logs',
                      'in_cls_weight_path': in_prefix + 'cls_weight.pt',
                      'in_cls_checkpoint_path': in_prefix + 'cls_checkpoint',
                      'out_cls_weight_path': out_prefix + 'cls_weight.pt',
                      'out_cls_checkpoint_path': out_prefix + 'cls_checkpoint'}

        config['checkpoint'] = checkpoint

        return config

    except IOError:
        sys.stderr.write('logging config file "%s" not found' % config_path)
        exit(-1)