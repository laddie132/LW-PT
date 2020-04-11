#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import torch
import torch.nn


class LinearMLC(torch.nn.Module):
    def __init__(self, model_config):
        super(LinearMLC, self).__init__()
        hidden_size = model_config['hidden_size']
        label_size = model_config['label_size']

        self.cls_layer = torch.nn.Linear(hidden_size * 4 * label_size, label_size)

    def forward(self, doc_rep):
        batch, label_size, _ = doc_rep.size()
        doc_rep = doc_rep.view(batch, -1)
        return torch.sigmoid(self.cls_layer(doc_rep))


class LabelGraphMLC(torch.nn.Module):
    def __init__(self, model_config):
        super(LabelGraphMLC, self).__init__()
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


class LabelWiseMLC(torch.nn.Module):
    def __init__(self, model_config):
        super(LabelWiseMLC, self).__init__()
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


class LabelGraphWiseMLC(torch.nn.Module):
    def __init__(self, model_config):
        super(LabelGraphWiseMLC, self).__init__()
        hidden_size = model_config['hidden_size']
        self.label_size = model_config['label_size']

        self.cls_layer = torch.nn.ModuleList([torch.nn.Linear(hidden_size * 4, 1)
                                              for _ in range(self.label_size)])
        self.label_graph = torch.nn.Parameter(torch.eye(self.label_size),
                                              requires_grad=True)

    def forward(self, doc_rep):
        doc_cls = []
        for i in range(self.label_size):
            doc_cls.append(self.cls_layer[i](doc_rep[:, i, :]))
        doc_cls = torch.cat(doc_cls, dim=-1)

        out_label = torch.sigmoid(doc_cls.mm(self.label_graph))
        return out_label