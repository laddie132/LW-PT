#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import torch
import torch.nn
import logging
from .base import BaseModule
from . import decoder
from . import e2e

logger = logging.getLogger(__name__)


class MultiLabelCls(BaseModule):
    def __init__(self, config):
        super(MultiLabelCls, self).__init__(
            config,
            name='cls',
            model=decoder.LinearMLC(
                input_size=config['model']['hidden_size'] * 4 * config['model']['label_size'],
                label_size=config['model']['label_size']))  # use doc-representation to classification

    @staticmethod
    def criterion(y_pred, y_true, reduction='mean'):
        """
        a personal negative log likelihood loss. It is useful to train a classification problem with `C` classes.
        :param y_pred: (batch, labels)
        :param y_true: (batch, labels), 0 or 1
        :param reduction:
        :return:
        """
        y_pred_log = torch.log(y_pred + 1e-8)   # to prevent Nan loss
        non_y_pred_log = torch.log(1 - y_pred + 1e-8)
        valid_docs_prob_log = y_pred_log * y_true.float()
        non_docs_prob_log = non_y_pred_log * (1 - y_true).float()
        batch_loss = -valid_docs_prob_log.sum(dim=-1) - non_docs_prob_log.sum(dim=-1)

        if reduction == 'none':
            return batch_loss
        elif reduction == 'sum':
            return batch_loss.sum()
        elif reduction == 'mean':
            return batch_loss.sum() / batch_loss.shape[0]
        else:
            raise ValueError(reduction)


class E2EMultiLabelCls(BaseModule):
    def __init__(self, config):
        super(E2EMultiLabelCls, self).__init__(
            config,
            name='pt',  # TODO: dynamically changed
            model=getattr(e2e, config['model']['name'])(config['model']))
