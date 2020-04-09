#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import torch
import torch.nn
from .base import BaseModule


class MultiCls(BaseModule):
    def __init__(self, config):
        super(MultiCls, self).__init__()
        self.name = 'cls'

        self.in_checkpoint_path = config['checkpoint']['in_cls_checkpoint_path']
        self.in_weight_path = config['checkpoint']['in_cls_weight_path']
        self.out_checkpoint_path = config['checkpoint']['out_cls_checkpoint_path']
        self.out_weight_path = config['checkpoint']['out_cls_weight_path']

        self.model = LabelGraphModel(config['model'])

    @staticmethod
    def criterion(y_pred, y_true, reduction='mean'):
        """
        a personal negative log likelihood loss. It is useful to train a classification problem with `C` classes.
        :param y_pred: (batch, labels)
        :param y_true: (batch, labels), 0 or 1
        :param reduction:
        :return:
        """
        y_pred_log = torch.log(y_pred)
        non_y_pred_log = torch.log(1 - y_pred)
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


class MultiClsModel(torch.nn.Module):
    def __init__(self, model_config):
        super(MultiClsModel, self).__init__()
        hidden_size = model_config['hidden_size']
        label_size = model_config['label_size']

        self.cls_layer = torch.nn.Linear(hidden_size * 4 * label_size, label_size)

    def forward(self, doc_rep):
        batch, label_size, _ = doc_rep.size()
        doc_rep = doc_rep.view(batch, -1)
        return torch.sigmoid(self.cls_layer(doc_rep))


class LabelGraphModel(torch.nn.Module):
    def __init__(self, model_config):
        super(LabelGraphModel, self).__init__()
        hidden_size = model_config['hidden_size']
        label_size = model_config['label_size']

        self.cls_layer = torch.nn.Linear(hidden_size * 4 * label_size, label_size)
        self.label_graph = torch.nn.Parameter(torch.eye(label_size),
                                              requires_grad=True)

    def forward(self, doc_rep):
        batch, label_size, _ = doc_rep.size()
        doc_rep = doc_rep.view(batch, -1)
        raw_label = self.cls_layer(doc_rep)
        out_label = torch.sigmoid(raw_label.mm(self.label_graph))

        return out_label


class LWClsModel(torch.nn.Module):
    def __init__(self, model_config):
        super(LWClsModel, self).__init__()
        hidden_size = model_config['hidden_size']
        self.label_size = model_config['label_size']

        self.cls_layer = torch.nn.ModuleList([torch.nn.Linear(hidden_size * 4, 1)
                                              for _ in range(self.label_size)])

    def forward(self, doc_rep):
        doc_sig = []
        for i in range(self.label_size):
            doc_sig.append(torch.sigmoid(self.cls_layer[i](doc_rep[:, i, :])))
        doc_sig = torch.cat(doc_sig, dim=-1)
        return doc_sig
