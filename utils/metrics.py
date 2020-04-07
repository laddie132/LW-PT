#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import torch
import torch.nn


def evaluate_acc(predict, truth):
    """
    compute the accuracy of predict value
    :param predict: (batch, _)
    :param truth: (batch)
    :return:
    """
    _, predict_max = predict.max(dim=1)

    batch_eq_num = torch.eq(predict_max, truth).long().sum().item()
    batch_acc = batch_eq_num / truth.shape[0]

    return batch_acc, batch_eq_num


def evaluate_acc_sigmoid(predict, truth):
    """
    accuracy evaluate with two classification on sigmoid
    :param predict: (batch,)
    :param truth: (batch,)
    :return:
    """
    predict_max = predict.gt(0.5).long()

    batch_eq_num = torch.eq(predict_max, truth).long().sum().item()
    batch_acc = batch_eq_num / truth.shape[0]

    return batch_acc, batch_eq_num


def evaluate_macro_f1(predict, truth):
    pass


def evaluate_micro_f1(predict, truth):
    pass
