#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import os
import logging
import torch
from utils.functions import save_model, load_checkpoint_parameters

logger = logging.getLogger(__name__)


class BaseModule(torch.nn.Module):
    def __init__(self, config, name, model):
        super(BaseModule, self).__init__()
        self.model = model

        self.in_checkpoint_path = config['checkpoint']['in_{}_checkpoint_path'.format(name)]
        self.in_weight_path = config['checkpoint']['in_{}_weight_path'.format(name)]
        self.out_checkpoint_path = config['checkpoint']['out_{}_checkpoint_path'.format(name)]
        self.out_weight_path = config['checkpoint']['out_{}_weight_path'.format(name)]

    def forward(self, *args):
        out = self.model(*args)
        return out

    def load_parameters(self, enable_cuda, force=False, strict=False):
        if force:
            assert os.path.exists(self.in_checkpoint_path)

        if os.path.exists(self.in_checkpoint_path):
            logger.info('loading parameters for {} module'.format(self.model.__class__.__name__))
            load_weight_path = load_checkpoint_parameters(self,
                                                          self.in_weight_path,
                                                          self.in_checkpoint_path,
                                                          enable_cuda,
                                                          strict)
            logger.info('loaded {} module from {}'.format(self.model.__class__.__name__, load_weight_path))

    def load_out_parameters(self, enable_cuda, force=False, strict=False):
        if force:
            assert os.path.exists(self.out_checkpoint_path)

        if os.path.exists(self.out_checkpoint_path):
            logger.info('loading parameters for {} module'.format(self.model.__class__.__name__))
            load_weight_path = load_checkpoint_parameters(self,
                                                          self.out_weight_path,
                                                          self.out_checkpoint_path,
                                                          enable_cuda,
                                                          strict)
            logger.info('loaded {} module from {}'.format(self.model.__class__.__name__, load_weight_path))

    def save_parameters(self, num):
        """
        Save the trained parameters
        :param num:
        :return:
        """
        logger.info('saving parameters for {} module on steps={}'.format(self.model.__class__.__name__, num))
        save_model(self,
                   num,
                   model_weight_path=self.out_weight_path + '-' + str(num),
                   checkpoint_path=self.out_checkpoint_path)
