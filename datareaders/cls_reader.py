#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""Dataset Reader for directly classification with e2e model"""

import pickle
import torch
import torch.utils.data
import logging
from .vocabulary import Vocabulary
from utils.functions import del_zeros_right, compute_mask

logger = logging.getLogger(__name__)


class DocClsReader:
    def __init__(self, config):
        self.num_workers = config['global']['num_data_workers']
        self.batch_size = config['train']['batch_size']
        self.data_path = config['dataset']['data_path']
        self.hierarchical = config['global']['hierarchical']

        self.data = {}
        self.meta_data = {}

        self.load_data()

    def load_data(self):
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        for name, value in data.items():
            self.data[name] = torch.tensor(value)

    def get_dataloader_train(self):
        return self._get_dataloader(self.data['x_train'], self.data['y_train'], shuffle=True)

    def get_dataloader_valid(self):
        return self._get_dataloader(self.data['x_valid'], self.data['y_valid'], shuffle=False)

    def get_dataloader_test(self):
        return self._get_dataloader(self.data['x_test'], self.data['y_test'], shuffle=False)

    def _get_dataloader(self, docs_rep, label, shuffle):
        doc_dataset = DocClsDataset(docs_rep, label, self.hierarchical)
        return torch.utils.data.DataLoader(doc_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=shuffle,
                                           collate_fn=doc_dataset.collect_fun)


class DocClsDataset(torch.utils.data.Dataset):
    def __init__(self, docs, labels, hierarchical):
        super(DocClsDataset, self).__init__()
        self.docs = docs
        self.labels = labels
        self.hierarchical = hierarchical

    def __len__(self):
        return self.docs.shape[0]

    def __getitem__(self, index):
        return self.docs[index], self.labels[index]

    def collect_fun(self, batch):
        docs = []
        labels = []

        for ele in batch:
            docs.append(ele[0])
            labels.append(ele[1])

        docs = torch.stack(docs, dim=0)
        labels = torch.stack(labels, dim=0)

        # compress on word level
        docs, _ = del_zeros_right(docs)
        docs_mask = compute_mask(docs, padding_idx=Vocabulary.PAD_IDX)

        # compress on sentence level
        if self.hierarchical:
            _, sent_right_idx = del_zeros_right(docs_mask.sum(-1))
            docs = docs[:, :sent_right_idx, :]
            docs_mask = docs_mask[:, :sent_right_idx, :]

        # logger.info('tar_d: {}, {}'.format(docs.dtype, docs.shape))
        # logger.info('label: {}, {}'.format(labels.dtype, labels.shape))

        return docs, docs_mask, labels
