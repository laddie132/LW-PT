#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""Dataset Reader"""

import h5py
import pickle
import random
import torch
import torch.utils.data
import logging
from .vocabulary import Vocabulary
from utils.functions import del_zeros_right, compute_mask

logger = logging.getLogger(__name__)


class QTReader:
    def __init__(self, config):
        self.config = config
        self.num_workers = self.config['global']['num_data_workers']
        self.cand_doc_size = self.config['global']['cand_doc_size']
        self.batch_size = self.config['train']['batch_size']
        self.train_iters = self.config['train']['train_iters']
        self.valid_iters = self.config['train']['valid_iters']

        self.data = {}
        self.meta_data = {}

        self.load_data()

    def load_h5_data(self):
        h5_path = self.config['dataset']['h5_path']
        with h5py.File(h5_path, 'r') as f:
            f_data = f['data']
            for name, value in f_data.items():
                self.data[name] = torch.tensor(value, dtype=torch.long)

            # f_meta_data = f['meta_data']
            # for name, value in f_meta_data.items():
            #     self.meta_data[name] = np.array[value]

    def load_data(self):
        with open(self.config['dataset']['data_path'], 'rb') as f:
            data = pickle.load(f)
        for name, value in data.items():
            self.data[name] = torch.tensor(value)

    def get_dataloader_train(self):
        return self._get_dataloader(self.data['x_train'], self.data['y_train'], self.train_iters)

    def get_dataloader_valid(self):
        return self._get_dataloader(self.data['x_valid'], self.data['y_valid'], self.valid_iters)

    def _get_dataloader(self, x, y, iters):
        doc_dataset = QTDataset(x, y, self.cand_doc_size)
        cur_sampler = torch.utils.data.sampler.RandomSampler(doc_dataset,
                                                             replacement=True,
                                                             num_samples=iters * self.batch_size)
        return torch.utils.data.DataLoader(doc_dataset,
                                           batch_size=self.batch_size,
                                           sampler=cur_sampler,
                                           num_workers=self.num_workers,
                                           collate_fn=QTDataset.collect_fun)


class QTDataset(torch.utils.data.Dataset):
    def __init__(self, docs, labels, cand_doc_size):
        super(QTDataset, self).__init__()
        self.docs = docs
        self.labels = labels

        self.cand_doc_size = cand_doc_size
        self.nums, self.max_sent_num, self.max_sent_len = self.docs.shape

        self.same_docs, self.diff_docs = self.gen_same_diff_docs()

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        cur_doc = self.docs[index]

        # random select a label as input
        cur_label = self.labels[index]
        qt_label_idx = random.choice(cur_label.nonzero().squeeze(-1).tolist())
        qt_label = torch.zeros(self.labels.shape[1], dtype=torch.long)
        qt_label[qt_label_idx] = 1

        # random generate candidates
        cand_docs_idx = random.sample(self.diff_docs[qt_label_idx], self.cand_doc_size - 1)
        tar_doc_idx = index
        while tar_doc_idx == index:
            tar_doc_idx = random.choice(self.same_docs[qt_label_idx])

        cand_docs_idx.append(tar_doc_idx)
        random.shuffle(cand_docs_idx)
        gt_idx = cand_docs_idx.index(tar_doc_idx)

        cand_docs = self.docs[cand_docs_idx]

        return cur_doc, cand_docs, qt_label, gt_idx

    def gen_same_diff_docs(self):
        label_size = self.labels.shape[1]
        same_docs = []
        diff_docs = []
        for i in range(label_size):
            same_idx = self.labels[:, i].nonzero().squeeze(-1).tolist()
            non_idx = (1 - self.labels[:, i]).nonzero().squeeze(-1).tolist()
            same_docs.append(same_idx)
            diff_docs.append(non_idx)
        return same_docs, diff_docs

    @staticmethod
    def collect_fun(batch):
        tar_d = []
        cand_ds = []
        qt_label = []
        gt_idx = []

        for ele in batch:
            tar_d.append(ele[0])
            cand_ds.append(ele[1])
            qt_label.append(ele[2])
            gt_idx.append(ele[3])

        tar_d = torch.stack(tar_d, dim=0)
        cand_ds = torch.stack(cand_ds, dim=0)
        qt_label = torch.stack(qt_label, dim=0)
        gt_idx = torch.tensor(gt_idx, dtype=torch.long)

        # compress on word level
        tar_d, _ = del_zeros_right(tar_d)
        tar_mask = compute_mask(tar_d, padding_idx=Vocabulary.padding_idx)

        cand_ds, _ = del_zeros_right(cand_ds)
        cand_mask = compute_mask(cand_ds, padding_idx=Vocabulary.padding_idx)

        # compress on sentence level
        _, sent_right_idx = del_zeros_right(tar_mask.sum(-1))
        tar_d = tar_d[:, :sent_right_idx, :]
        tar_mask = tar_mask[:, :sent_right_idx, :]

        _, sent_right_idx = del_zeros_right(cand_mask.sum(-1))
        cand_ds = cand_ds[:, :, :sent_right_idx, :]
        cand_mask = cand_mask[:, :, :sent_right_idx, :]

        # logger.info('tar_d: {}, {}'.format(tar_d.dtype, tar_d.shape))
        # logger.info('cand_ds: {}, {}'.format(cand_ds.dtype, cand_ds.shape))
        # logger.info('label: {}, {}'.format(qt_label.dtype, qt_label.shape))

        return tar_d, tar_mask, cand_ds, cand_mask, qt_label, gt_idx
