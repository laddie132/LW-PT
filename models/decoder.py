#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import torch
import torch.nn


class LinearMLC(torch.nn.Module):
    def __init__(self, input_size, label_size):
        super(LinearMLC, self).__init__()
        self.cls_layer = torch.nn.Linear(input_size, label_size)

    def forward(self, doc_rep):
        batch = doc_rep.size(0)
        # doc_rep = doc_rep.view(batch, label_size, -1, 2)[:, :, :, 1].contiguous()
        doc_rep = doc_rep.view(batch, -1)
        return torch.sigmoid(self.cls_layer(doc_rep))


class TwoLinearMLC(torch.nn.Module):
    def __init__(self, input_size, hidden_size, label_size):
        super(TwoLinearMLC, self).__init__()
        self.hidden_layer = torch.nn.Linear(input_size, hidden_size)
        self.cls_layer = torch.nn.Linear(hidden_size, label_size)

    def forward(self, doc_rep):
        batch = doc_rep.size(0)
        # doc_rep = doc_rep.view(batch, label_size, -1, 2)[:, :, :, 1].contiguous()
        doc_rep = doc_rep.view(batch, -1)
        h = torch.relu(self.hidden_layer(doc_rep))
        return torch.sigmoid(self.cls_layer(h))


class LabelGraphMLC(torch.nn.Module):
    def __init__(self, input_size, label_size):
        super(LabelGraphMLC, self).__init__()
        self.cls_layer = torch.nn.Linear(input_size, label_size)
        self.label_graph = torch.nn.Parameter(torch.eye(label_size),
                                              requires_grad=True)

    def forward(self, doc_rep):
        batch = doc_rep.size(0)
        doc_rep = doc_rep.view(batch, -1)
        raw_label = self.cls_layer(doc_rep)
        out_label = torch.sigmoid(raw_label.mm(self.label_graph))

        return out_label


class LabelWiseMLC(torch.nn.Module):
    def __init__(self, input_size, label_size):
        super(LabelWiseMLC, self).__init__()
        self.cls_layer = torch.nn.Linear(input_size, label_size)

    def forward(self, doc_rep):
        batch, label_size, _ = doc_rep.size()
        doc_label = self.cls_layer(doc_rep)     # (batch, label_size, label_size)
        select_idx = torch.eye(label_size, device=doc_rep.device).repeat(batch, 1, 1).bool()
        doc_label = doc_label[select_idx].view(batch, label_size)

        doc_sig = torch.sigmoid(doc_label)
        return doc_sig


class LabelWiseGraphMLC(torch.nn.Module):
    def __init__(self, input_size, label_size):
        super(LabelWiseGraphMLC, self).__init__()
        self.cls_layer = torch.nn.Linear(input_size, label_size)
        self.label_graph = torch.nn.Parameter(torch.eye(label_size),
                                              requires_grad=True)

    def forward(self, doc_rep):
        batch, label_size, _ = doc_rep.size()
        doc_label = self.cls_layer(doc_rep)  # (batch, label_size, label_size)
        select_idx = torch.eye(label_size, device=doc_rep.device).repeat(batch, 1, 1).bool()
        doc_label = doc_label[select_idx].view(batch, label_size)

        out_label = torch.sigmoid(doc_label.mm(self.label_graph))
        return out_label
