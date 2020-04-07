#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""Dataset Reader for document representation"""

import h5py
import pickle
import torch
import torch.utils.data
import logging
from utils.functions import hierarchical_sequence_mask

logger = logging.getLogger(__name__)


class QTRepReader:
    def __init__(self, config):
        self.config = config
        self.num_workers = self.config['global']['num_data_workers']
        self.batch_size = self.config['train']['batch_size']

        self.data = {}
        self.meta_data = {}

        self.load_data()

    def load_data(self):
        with open(self.config['dataset']['data_path'], 'rb') as f:
            data = pickle.load(f)
        for name, value in data.items():
            self.data[name] = torch.tensor(value)

    def get_dataloader_train(self):
        return self._get_dataloader(self.data['x_train'], self.data['seq_train'])

    def get_dataloader_valid(self):
        return self._get_dataloader(self.data['x_valid'], self.data['seq_valid'])

    def get_dataloader_test(self):
        return self._get_dataloader(self.data['x_test'], self.data['seq_test'])

    def _get_dataloader(self, x, seq):
        doc_dataset = QTRepDataset(x, seq)
        return torch.utils.data.DataLoader(doc_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           collate_fn=QTRepDataset.collect_fun)


class QTRepDataset(torch.utils.data.Dataset):
    def __init__(self, docs, seq_len):
        super(QTRepDataset, self).__init__()
        self.docs = docs
        self.seq_len = seq_len

        self.nums, self.max_sent_num, self.max_sent_len, _ = self.docs.shape

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        cur_doc = self.docs[index]
        cur_len = self.seq_len[index]

        return cur_doc, cur_len

    @staticmethod
    def collect_fun(batch):
        cur_doc = []
        cur_len = []

        for ele in batch:
            cur_doc.append(ele[0])
            cur_len.append(ele[1])

        cur_doc = torch.stack(cur_doc, dim=0).float()
        cur_len = torch.stack(cur_len, dim=0).long()

        # generate cur mask
        cur_mask = hierarchical_sequence_mask(cur_len)
        _, cur_sent_num, cur_sent_len = cur_mask.size()
        cur_doc = cur_doc[:, :cur_sent_num, :cur_sent_len, :]

        return cur_doc, cur_mask
