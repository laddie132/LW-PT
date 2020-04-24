#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import json
import torch


def get_gt_labels(gt_path):
    with open(gt_path, 'r') as f:
        data = json.load(f)
    gt_labels = []

    for ele in data:
        labels = ele['tags']
        labels.sort()
        gt_labels.append({'name': ele['name'], 'labels': labels})
    return gt_labels


def predict_to_json(pred_path, meta_path, gt_path, save_path):
    pred_prob = torch.load(pred_path)
    pred = pred_prob.gt(0.5).int().tolist()

    with open(meta_path, 'r') as f:
        labels = json.load(f)['labels']

    gt_labels = get_gt_labels(gt_path)

    assert len(gt_labels) == len(pred)

    for plist, gt_d in zip(pred, gt_labels):
        plabels = [labels[i] for i, p in enumerate(plist) if p > 0]
        gt_d['predict'] = plabels

    with open(save_path, 'w') as wf:
        json.dump(gt_labels, wf, indent=2, ensure_ascii=False)


def combine_predicts(pred_path_lst, name_lst, save_path):
    pred_lst = []
    for p in pred_path_lst:
        with open(p, 'r') as f:
            pred_lst.append(json.load(f))

    pred_size = len(pred_lst)
    assert pred_size > 0

    data_size = len(pred_lst[0])
    assert data_size > 0

    cb_data = []
    for i in range(data_size):
        cur = {'name': pred_lst[0][i]['name'], 'labels': pred_lst[0][i]['labels']}
        for j in range(pred_size):
            cur[name_lst[j]] = pred_lst[j][i]['predict']
        cb_data.append(cur)

    with open(save_path, 'w') as wf:
        json.dump(cb_data, wf, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    predict_to_json(pred_path='outputs/rmsc/qt-hlwan-ln-c3-8k-ft-mlp-20-test-0.8/predict.pt',
                    meta_path='data/rmsc.meta.json',
                    gt_path='data/rmsc/rmsc.data.test.json',
                    save_path='outputs/rmsc/qt-hlwan-ln-c3-8k-ft-mlp-20-test-0.8/predict.json')

    predict_to_json(pred_path='outputs/rmsc/qt-hlwan-ln-c3-8k-mlp-20-test-0.9/predict.pt',
                    meta_path='data/rmsc.meta.json',
                    gt_path='data/rmsc/rmsc.data.test.json',
                    save_path='outputs/rmsc/qt-hlwan-ln-c3-8k-mlp-20-test-0.9/predict.json')

    predict_to_json(pred_path='outputs/rmsc/hlwan-cls-20/predict.pt',
                    meta_path='data/rmsc.meta.json',
                    gt_path='data/rmsc/rmsc.data.test.json',
                    save_path='outputs/rmsc/hlwan-cls-20/predict.json')

    combine_predicts(pred_path_lst=['outputs/rmsc/hlwan-cls-20/predict.json',
                                    'outputs/rmsc/qt-hlwan-ln-c3-8k-mlp-20-test-0.9/predict.json',
                                    'outputs/rmsc/qt-hlwan-ln-c3-8k-ft-mlp-20-test-0.8/predict.json'],
                     name_lst=['hlwan', 'hlwan-qt', 'hlwan-qt-ft'],
                     save_path='data/rmsc_case_study.json')
