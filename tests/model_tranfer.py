#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import os
import sys
sys.path.append(os.getcwd())

import argparse
import torch
import logging
from collections import OrderedDict
from models import *
from utils.config import init_logging, read_config

init_logging()
logger = logging.getLogger(__name__)


def transform(pre_model_path, tar_model_path, cur_model):
    pre_weight = torch.load(pre_model_path, map_location=lambda storage, loc: storage)
    pre_keys = pre_weight.keys()
    pre_value = pre_weight.values()

    cur_weight = cur_model.state_dict()
    del cur_weight['model.embedding_layer.weight']
    cur_keys = cur_weight.keys()

    assert len(pre_keys) == len(cur_keys)
    logging.info('pre-keys: ' + str(pre_keys))
    logging.info('cur-keys: ' + str(cur_keys))

    new_weight = OrderedDict(zip(cur_keys, pre_value))
    torch.save(new_weight, tar_model_path)


def main(pre_model_path, tar_model_path):
    logger.info('loading config file...')
    config = read_config('config/rmsc.yaml')

    logger.info('constructing model...')
    model = LWPT(config)

    logging.info("transforming model from '%s' to '%s'..." % (pre_model_path, tar_model_path))
    transform(pre_model_path, tar_model_path, model)

    logging.info('finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="transform a old model weight to the newest network")
    parser.add_argument('--input', '-i', required=True, type=str, dest='pre_weight')
    parser.add_argument('--output', '-o', required=True, type=str, dest='tar_weight')

    args = parser.parse_args()
    main(args.pre_weight, args.tar_weight)