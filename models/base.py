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
    def __init__(self):
        super(BaseModule, self).__init__()
        self.name = 'base'
        self.model = None

        self.in_checkpoint_path = None
        self.in_weight_path = None
        self.out_checkpoint_path = None
        self.out_weight_path = None

    def forward(self, *args, e2e=True):
        if e2e:
            out = self.model(*args)
        else:
            with torch.no_grad():
                out = self.model(*args)
        return out

    def load_parameters(self, enable_cuda, force=False, strict=False):
        if force:
            assert os.path.exists(self.in_checkpoint_path)

        if os.path.exists(self.in_checkpoint_path):
            logger.info('loading parameters for {} module'.format(self.name))
            load_weight_path = load_checkpoint_parameters(self,
                                                          self.in_weight_path,
                                                          self.in_checkpoint_path,
                                                          enable_cuda,
                                                          strict)
            logger.info('loaded {} module from {}'.format(self.name, load_weight_path))

    def save_parameters(self, num):
        """
        Save the trained parameters
        :param num:
        :return:
        """
        logger.info('saving parameters for {} module on steps={}'.format(self.name, num))
        save_model(self,
                   num,
                   model_weight_path=self.out_weight_path + '-' + str(num),
                   checkpoint_path=self.out_checkpoint_path)