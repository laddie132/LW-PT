#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""Analysis labels F1 score with labels frequency on RMSC & AAPD dataset"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()


def count_rmsc_label(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)

    label_cnt = {}
    all_cnt = 0

    for song in data:
        all_cnt += 1
        for t in song['tags']:
            label_cnt[t] = label_cnt.get(t, 0) + 1

    return label_cnt, all_cnt


def count_aapd_label(data_path):
    label_cnt = {}
    all_cnt = 0

    with open(data_path, 'r') as f:
        for line in f:
            label = line.strip().split()
            if len(label) > 0:
                all_cnt += 1

                for t in label:
                    label_cnt[t] = label_cnt.get(t, 0) + 1
    return label_cnt, all_cnt


def count_label_freq(data_prefix, labels, dtype):
    if dtype == 'rmsc':
        train_label_cnt, train_cnt = count_rmsc_label(data_prefix + '/rmsc.data.train.json')
        valid_label_cnt, valid_cnt = count_rmsc_label(data_prefix + '/rmsc.data.valid.json')
        test_label_cnt, test_cnt = count_rmsc_label(data_prefix + '/rmsc.data.test.json')
    elif dtype == 'aapd':
        train_label_cnt, train_cnt = count_aapd_label(data_prefix + '/label_train')
        valid_label_cnt, valid_cnt = count_aapd_label(data_prefix + '/label_val')
        test_label_cnt, test_cnt = count_aapd_label(data_prefix + '/label_test')
    else:
        raise ValueError(dtype)

    label_freq = {}
    all_cnt = train_cnt + valid_cnt + test_cnt
    for k in labels:
        label_freq[k] = (train_label_cnt.get(k, 0) + valid_label_cnt.get(k, 0) + test_label_cnt.get(k, 0)) / all_cnt

    return label_freq


def discrete_freq(sort_freq, sort_three_f1, sep, minv, maxv):
    f1_len = len(sort_three_f1)

    # split to discrete
    dis_freq = [i / 100 for i in range(minv, maxv, sep)]
    dis_three_f1 = [[0 for _ in range(minv, maxv, sep)] for _ in range(f1_len)]
    dis_three_cnt = [[0 for _ in range(minv, maxv, sep)] for _ in range(f1_len)]

    for i, freq in enumerate(sort_freq):
        idx = min(int(freq * 100 / sep), len(dis_freq) - 1)

        for j, sort_f1 in enumerate(sort_three_f1):
            dis_three_f1[j][idx] += sort_f1[i]
            dis_three_cnt[j][idx] += 1

    x_freq = []
    y_three_f1 = [[] for _ in range(f1_len)]
    for i, freq in enumerate(dis_freq):
        if dis_three_cnt[0][i] > 0:
            x_freq.append(freq)
        else:
            continue

        for j, dis_f1 in enumerate(dis_three_f1):
            y_three_f1[j].append(dis_f1[i] / dis_three_cnt[j][i])
    return x_freq, y_three_f1


def show_labels(data_prefix, meta_path, metrics_path, names, out_path, dtype):
    with open(meta_path, 'r') as f:
        labels = json.load(f)['labels']
        labels = np.array(labels)

    three_f1 = []
    for p in metrics_path:
        with open(p, 'r') as f:
            f1 = json.load(f)['label_f1']
            three_f1.append(np.array(f1))
    macro_three_f1 = [x.mean() for x in three_f1]

    label_freq = count_label_freq(data_prefix, labels, dtype)
    freq = np.array([label_freq[k] for k in labels])

    sort_idx = freq.argsort()
    sort_freq = freq[sort_idx]
    sort_labels = labels[sort_idx]
    sort_three_f1 = [f1[sort_idx] for f1 in three_f1]

    dis_freq, dis_three_f1 = discrete_freq(sort_freq, sort_three_f1,
                                           minv=2, maxv=35, sep=2)

    # labels_f1 = dict(zip(labels, f1))
    # print(labels_f1)
    # print(label_freq)

    plt.figure()
    plt.scatter(dis_freq, dis_three_f1[0], marker='^', label=names[0])
    plt.scatter(dis_freq, dis_three_f1[1], marker='d', label=names[1])
    plt.scatter(dis_freq, dis_three_f1[2], marker='p', label=names[2])

    # plt.plot([0.02, 0.33], [macro_three_f1[0], macro_three_f1[0]], lw=0.7, ls='--')
    # plt.plot([0.02, 0.33], [macro_three_f1[1], macro_three_f1[1]], lw=0.7, ls='--')
    # plt.plot([0.02, 0.33], [macro_three_f1[2], macro_three_f1[2]], lw=0.7, ls='--')

    plt.xlabel('Frequency of label')
    plt.ylabel('Average F1 score(+)')
    # plt.title('Labels F1(+)')

    plt.xticks([x / 100 for x in range(4, 35, 4)])
    plt.yticks([x / 10 for x in range(1, 10)])
    plt.legend(loc='lower right')

    plt.savefig(out_path)
    # plt.show()


if __name__ == '__main__':
    show_labels(data_prefix='data/rmsc/',
                meta_path='data/rmsc.meta.json',
                metrics_path=['outputs/rmsc/hlwan-mlp-20/metrics.json',
                              'outputs/rmsc/qt-hlwan-ln-c3-8k-mlp-20/metrics.json',
                              'outputs/rmsc/qt-hlwan-ln-c3-8k-ft-mlp-20/metrics.json'],
                names=['HLW-LSTM', 'HLW-LSTM+PT', 'HLW-LSTM+PT+FT'],
                out_path='data/rmsc_labels_freq_f1.eps',
                dtype='rmsc')

    show_labels(data_prefix='data/aapd/',
                meta_path='data/aapd.meta.json',
                metrics_path=['outputs/aapd/lwlstm-2-mlp-20/metrics.json',
                              'outputs/aapd/qt-lwlstm-2-256-ln-c3-3k-mlp/metrics.json',
                              'outputs/aapd/qt-lwlstm-2-256-ln-c3-3k-ft-mlp/metrics.json'],
                names=['LW-LSTM', 'LW-LSTM+PT', 'LW-LSTM+PT+FT'],
                out_path='data/aapd_labels_freq_f1.eps',
                dtype='aapd')
