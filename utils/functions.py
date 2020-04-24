#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import os
import torch
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.optim as optim
import torch.cuda


def set_seed(seed):
    """
    set random seed for re-implement
    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """
    FROM KERAS
    Pads each sequence to the same length:
    the length of the longest sequence.
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def to_long_tensor(np_array):
    """
    convert to long torch tensor
    :param np_array:
    :return:
    """
    return torch.as_tensor(np_array, dtype=torch.int64)


def to_float_tensor(np_array):
    """
    convert to long torch tensor
    :param np_array:
    :return:
    """
    return torch.as_tensor(np_array, dtype=torch.float)


def del_zeros_right(tensor):
    """
    delete the extra zeros in the right column
    :param tensor: (*, seq_len)
    :return:
    """
    seq_len = tensor.size()[-1]
    flip_tensor = tensor.view(-1, seq_len)

    last_col = seq_len
    for i in range(seq_len - 1, -1, -1):
        tmp_col = flip_tensor[:, i]
        tmp_sum_col = torch.sum(tmp_col).item()
        if tmp_sum_col > 0:
            break

        last_col = i

    flip_tensor = flip_tensor[:, :last_col]

    rtn_size = list(tensor.size()[:-1]) + [-1]
    rtn_tensor = flip_tensor.view(rtn_size)
    return rtn_tensor, last_col


def count_parameters(model):
    """
    get parameters count that require grad
    :param model:
    :return:
    """
    parameters_num = 0
    for par in model.parameters():
        if not par.requires_grad:
            continue

        tmp_par_shape = par.size()
        tmp_par_size = 1
        for ele in tmp_par_shape:
            tmp_par_size *= ele
        parameters_num += tmp_par_size
    return parameters_num


def compute_mask(v, padding_idx=0):
    """
    compute mask on given tensor v
    :param v:
    :param padding_idx:
    :return:
    """
    mask = torch.ne(v, padding_idx).float()
    return mask


def generate_mask(batch_length):
    """
    deprecated by `sequence_mask`
    generate mask with given length(should be sorted) of each element in batch
    :param batch_length: tensor
    :return:
    """
    sum_one = torch.sum(batch_length)
    one = torch.ones(sum_one.item())

    mask_packed = torch.nn.utils.rnn.PackedSequence(one, batch_length)
    mask, _ = torch.nn.utils.rnn.pad_packed_sequence(mask_packed)

    return mask


def compute_top_layer_mask(sub_layer_mask):
    """
    compute top layer mask by sub layer mask
    :param sub_layer_mask: (batch, top_len, sub_len)
    :return:
    """

    sub_sum = torch.sum(sub_layer_mask, dim=-1)
    top_mask = torch.gt(sub_sum, 0).int()
    return top_mask


def sequence_mask(lengths, max_len=None):
    """
    forked from https://github.com/baidu/knowledge-driven-dialogue/blob/master/generative_pt/source/utils/misc.py
    Creates a boolean mask from sequence lengths.
    """
    if max_len is None:
        max_len = lengths.max().item()
    mask = torch.arange(0, max_len, dtype=torch.long).type_as(lengths)
    mask = mask.unsqueeze(0)
    mask = mask.repeat(1, *lengths.size(), 1)
    mask = mask.squeeze(0)
    mask = mask.lt(lengths.unsqueeze(-1))
    #mask = mask.repeat(*lengths.size(), 1).lt(lengths.unsqueeze(-1))
    return mask


def hierarchical_sequence_mask(lengths):
    """
    Creates a boolean mask from hierarchical sequence lengths.
    """
    batch, s = lengths.size()
    mask = sequence_mask(lengths.view(-1), max_len=None)
    mask = mask.view(batch, s, -1)

    # top layer mask
    top_mask = compute_top_layer_mask(mask)
    _, max_s_len = del_zeros_right(top_mask)

    mask = mask[:, :max_s_len, :]
    return mask


def masked_softmax(x, m=None, dim=-1):
    """
    Softmax with mask
    :param x:
    :param m:
    :param dim:
    :return:
    """
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=dim, keepdim=True) + 1e-6)
    return softmax


def load_model_parameters(model, weight_path, enable_cuda=False, strict=False, replace=()):
    weight = torch.load(weight_path, map_location=lambda storage, loc: storage)
    if enable_cuda:
        weight = torch.load(weight_path, map_location=lambda storage, loc: storage.cuda())

    if len(replace) == 2:
        cur_keys = [k.replace(replace[0], replace[1]) for k in weight.keys()]
        weight = OrderedDict(zip(cur_keys, weight.values()))
    model.load_state_dict(weight, strict=strict)


def load_checkpoint_parameters(model, weight_path_prefix, checkpoint_path,
                               enable_cuda=False, strict=False, replace=()):
    with open(checkpoint_path, 'r') as f:
        checkpoint = f.readlines()[0].strip()
    load_weight_path = weight_path_prefix + '-' + checkpoint

    assert os.path.exists(load_weight_path)
    load_model_parameters(model, load_weight_path, enable_cuda, strict, replace)

    return load_weight_path


def save_model(model, num, model_weight_path, checkpoint_path):
    """
    save model weight without embedding
    :param model:
    :param num:
    :param model_weight_path:
    :param checkpoint_path:
    :return:
    """
    # save model weight
    model_weight = model.state_dict()

    torch.save(model_weight, model_weight_path)

    with open(checkpoint_path, 'w') as checkpoint_f:
        checkpoint_f.write('%d' % num)


def show_cuda_memory():
    mem_allcated = torch.cuda.memory_allocated() / 1024 / 1024
    mem_cached = torch.cuda.memory_cached() / 1024 / 1024

    print('allcated: %d MB \t cached: %d MB' % (mem_allcated, mem_cached))


def draw_heatmap_sea(x, title, save_path, xlabels="auto", ylabels="auto", cmap='Blues', center=None, vmax=None, vmin=None,
                     inches=(21, 2), linewidths=0, left=None, bottom=None, right=None, top=None):
    """
    draw matrix heatmap with seaborn
    :param x:
    :param xlabels:
    :param ylabels:
    :param cmap: YlGn, YlGnBu, YlOrBr, YlOrRd, PuBuGn, PuRd
    :param save_path:
    :param inches:
    :param bottom:
    :param linewidths:
    :return:
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    plt.title(title)
    sns.heatmap(x, linewidths=linewidths, ax=ax, cmap=cmap, xticklabels=xlabels, yticklabels=ylabels,
                center=center, vmax=vmax, vmin=vmin)
    fig.set_size_inches(inches)
    fig.savefig(save_path)
