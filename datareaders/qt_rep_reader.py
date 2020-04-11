#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""Dataset Reader for document representation"""

import pickle
import torch
import torch.utils.data
import logging
from .vocabulary import Vocabulary
from utils.functions import del_zeros_right, compute_mask

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
            self.data[name] = torch.tensor(value, dtype=torch.long)

    def get_dataloader_train(self):
        return self._get_dataloader(self.data['x_train'])

    def get_dataloader_valid(self):
        return self._get_dataloader(self.data['x_valid'])

    def get_dataloader_test(self):
        return self._get_dataloader(self.data['x_test'])

    def _get_dataloader(self, x):
        doc_dataset = QTRepDataset(x)
        return torch.utils.data.DataLoader(doc_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           collate_fn=QTRepDataset.collect_fun)


class QTRepDataset(torch.utils.data.Dataset):
    def __init__(self, docs):
        super(QTRepDataset, self).__init__()
        self.docs = docs

        self.nums, self.max_sent_num, self.max_sent_len = self.docs.shape

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        cur_doc = self.docs[index]

        return cur_doc

    @staticmethod
    def collect_fun(doc):
        doc = torch.stack(doc, dim=0)

        # compress on word level
        doc, _ = del_zeros_right(doc)
        doc_mask = compute_mask(doc, padding_idx=Vocabulary.PAD_IDX)

        # compress on sentence level
        _, sent_right_idx = del_zeros_right(doc_mask.sum(-1))
        doc = doc[:, :sent_right_idx, :]
        doc_mask = doc_mask[:, :sent_right_idx, :]

        return doc, doc_mask
