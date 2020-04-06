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
from utils.functions import hierarchical_sequence_mask

logger = logging.getLogger(__name__)


class QTReader:
    def __init__(self, config):
        self.config = config
        self.num_workers = self.config['global']['num_data_workers']
        self.cand_doc_size = self.config['global']['cand_doc_size']
        self.batch_size = self.config['train']['batch_size']
        self.train_iters = self.config['train']['train_iters']
        self.test_iters = self.config['train']['test_iters']

        self.data = {}
        self.meta_data = {}

        self.load_data()

    def load_data(self):
        h5_path = self.config['data']['h5_path']
        with h5py.File(h5_path, 'r') as f:
            f_data = f['data']
            for name, value in f_data.items():
                self.data[name] = torch.tensor(value)

            # f_meta_data = f['meta_data']
            # for name, value in f_meta_data.items():
            #     self.meta_data[name] = np.array[value]

    def get_dataloader_train(self):
        return self._get_dataloader(self.train_iters)

    def get_dataloader_test(self):
        return self._get_dataloader(self.test_iters)

    def _get_dataloader(self, iters):
        doc_dataset = QTDataset(self.data['x_train'], self.data['y_train'], self.data['seq_train'],
                                self.cand_doc_size)
        cur_sampler = torch.utils.data.sampler.RandomSampler(doc_dataset,
                                                             replacement=True,
                                                             num_samples=iters * self.batch_size)
        return torch.utils.data.DataLoader(doc_dataset,
                                           batch_size=self.batch_size,
                                           sampler=cur_sampler,
                                           num_workers=self.num_workers,
                                           collate_fn=QTDataset.collect_fun)


class QTDataset(torch.utils.data.Dataset):
    def __init__(self, docs, labels, seq_len, cand_doc_size):
        super(QTDataset, self).__init__()
        self.docs = docs
        self.labels = labels
        self.seq_len = seq_len

        self.cand_doc_size = cand_doc_size
        self.nums, self.max_sent_num, self.max_sent_len, _ = self.docs.shape

        self.same_docs, self.diff_docs = self.gen_same_diff_docs()

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        cur_doc = self.docs[index]
        cur_len = self.seq_len[index]

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
        cand_len = self.seq_len[cand_docs_idx]

        return cur_doc, cur_len, cand_docs, cand_len, qt_label, gt_idx

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
        cur_doc = []
        cur_len = []
        cand_docs = []
        cand_len = []
        qt_label = []
        gt_idx = []

        for ele in batch:
            cur_doc.append(ele[0])
            cur_len.append(ele[1])
            cand_docs.append(ele[2])
            cand_len.append(ele[3])
            qt_label.append(ele[4])
            gt_idx.append(ele[5])

        cur_doc = torch.stack(cur_doc, dim=0)
        cur_len = torch.stack(cur_len, dim=0)
        cand_docs = torch.stack(cand_docs, dim=0)
        cand_len = torch.stack(cand_len, dim=0)
        qt_label = torch.stack(qt_label, dim=0)
        gt_idx = torch.tensor(gt_idx, dtype=torch.long)

        # generate cur mask
        cur_mask = hierarchical_sequence_mask(cur_len)
        _, cur_sent_num, cur_sent_len = cur_mask.size()
        cur_doc = cur_doc[:, :cur_sent_num, :cur_sent_len, :]

        # generate cand mask
        batch, cand_size, max_sent_num = cand_len.size()
        cand_len = cand_len.view(-1, max_sent_num)
        cand_mask = hierarchical_sequence_mask(cand_len)

        _, sent_num, sent_len = cand_mask.size()
        cand_mask = cand_mask.view(batch, cand_size, sent_num, sent_len)
        cand_docs = cand_docs[:, :, :sent_num, :sent_len, :]

        return cur_doc, cur_mask, cand_docs, cand_mask, qt_label, gt_idx
