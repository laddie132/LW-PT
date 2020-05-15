#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""Dataset Reader for classification based on pre-trained representation"""

import pickle
import torch
import torch.utils.data
import logging

logger = logging.getLogger(__name__)


class DocRepClsReader:
    def __init__(self, config):
        self.num_workers = config['global']['num_data_workers']
        self.batch_size = config['train']['batch_size']
        self.data_path = config['dataset']['data_path']
        self.h5_path = config['dataset']['h5_path']
        self.doc_rep_path = config['dataset']['doc_rep_path']

        self.data = {}
        self.load_data()
        self.docs_rep = self.load_doc_rep()

    def load_data(self):
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        for name, value in data.items():
            self.data[name] = torch.tensor(value, dtype=torch.long)

    def load_doc_rep(self, on_cpu=True):
        """
        Load documents representation by pytorch binary file
        :param on_cpu:
        :return:
        """
        if on_cpu:
            docs_rep = torch.load(self.doc_rep_path, map_location=torch.device('cpu'))
        else:
            docs_rep = torch.load(self.doc_rep_path)
        return docs_rep

    def get_dataloader_train(self):
        return self._get_dataloader(self.docs_rep['train_doc_rep'], self.data['y_train'], shuffle=True)

    def get_dataloader_valid(self):
        return self._get_dataloader(self.docs_rep['valid_doc_rep'], self.data['y_valid'], shuffle=False)

    def get_dataloader_test(self):
        return self._get_dataloader(self.docs_rep['test_doc_rep'], self.data['y_test'], shuffle=False)

    def _get_dataloader(self, docs_rep, label, shuffle):
        doc_dataset = DocRepClsDataset(docs_rep, label)
        return torch.utils.data.DataLoader(doc_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=shuffle)


class DocRepClsDataset(torch.utils.data.Dataset):
    def __init__(self, docs_rep, label):
        super(DocRepClsDataset, self).__init__()
        self.docs_rep = docs_rep
        self.label = label
        batch, labels, _ = self.docs_rep.shape

    def __len__(self):
        return self.docs_rep.shape[0]

    def __getitem__(self, index):
        return self.docs_rep[index], self.label[index]