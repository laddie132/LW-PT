#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import os
import logging
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from .base import BaseDataset

logger = logging.getLogger(__name__)


class RCV1V2(BaseDataset):
    """
    RCV1-V2 dataset
    """

    def __init__(self, data_path, random_seed):
        super(RCV1V2, self).__init__(h5_path='data/rcv1v2.h5',
                                     save_data_path='data/rcv1v2.pkl',
                                     save_meta_data_path='data/rcv1v2.pkl.meta',
                                     w2v_path='data/rcv1v2_word2vec.model',
                                     load_emb=False,
                                     emb_dim=256,
                                     max_vocab_size=None,
                                     random_seed=random_seed)
        self.max_doc_length = 500
        self.label_size = 103

        self.data_path = 'data/rcv1-v2/sgm' if data_path is '' else data_path
        self.raw_texts_labels = {}

    def extract(self):
        logger.info('extracting dataset...')
        train_text_labels, train_word_sum, train_label_sum = \
            self.load_data(os.path.join(self.data_path, 'train.src.id'),
                           os.path.join(self.data_path, 'train.tgt.id'))
        val_text_labels, val_word_sum, val_label_sum = \
            self.load_data(os.path.join(self.data_path, 'valid.src.id'),
                           os.path.join(self.data_path, 'valid.tgt.id'))
        test_text_labels, test_word_sum, test_label_sum = \
            self.load_data(os.path.join(self.data_path, 'test.src.id'),
                           os.path.join(self.data_path, 'test.tgt.id'))

        self.raw_texts_labels['train'] = train_text_labels
        self.raw_texts_labels['valid'] = val_text_labels
        self.raw_texts_labels['test'] = test_text_labels

        data_size = len(train_text_labels) + len(val_text_labels) + len(test_text_labels)
        self.attrs['data_size'] = data_size
        self.attrs['train_size'] = len(train_text_labels)
        self.attrs['valid_size'] = len(val_text_labels)
        self.attrs['test_size'] = len(test_text_labels)
        logger.info('Size: {}'.format(data_size))
        logger.info('Train size: {}'.format(len(train_text_labels)))
        logger.info('Valid size: {}'.format(len(val_text_labels)))
        logger.info('Test size: {}'.format(len(test_text_labels)))

        ave_text_len = (train_word_sum + val_word_sum + test_word_sum) * 1.0 / data_size
        ave_label_size = (train_label_sum + val_label_sum + test_label_sum) * 1.0 / data_size
        logger.info('Ave text len: {}'.format(ave_text_len))
        logger.info('Ave label size: {}'.format(ave_label_size))
        self.attrs['ave_text_len'] = ave_text_len
        self.attrs['ave_label_size'] = ave_label_size

        return None, None

    def train_emb(self, total_docs, total_labels):
        pass

    def load_data(self, text_path, label_path):
        texts_labels = []
        texts_word_sum = 0
        labels_sum = 0

        with open(text_path, 'r') as tf, open(label_path, 'r') as lf:
            for text, label in zip(tf, lf):
                if text != '' and label != '':
                    text = list(map(lambda x: int(x), text.strip().split()))
                    label = list(map(lambda x: int(x), label.strip().split()))
                    label = label[1:-1]

                    labels_sum += len(label)
                    texts_word_sum += len(text)

                    texts_labels.append((text, label))

        return texts_labels, texts_word_sum, labels_sum

    def transform(self):
        x_train, y_train = self.t5_data(self.raw_texts_labels['train'])
        x_valid, y_valid = self.t5_data(self.raw_texts_labels['valid'])
        x_test, y_test = self.t5_data(self.raw_texts_labels['test'])

        data = {'x_train': x_train,
                'x_valid': x_valid,
                'x_test': x_test,
                'y_train': y_train,
                'y_valid': y_valid,
                'y_test': y_test}
        meta_data = {}
        return data, meta_data

    def t5_data(self, data):
        labels_origin = []
        data_size = len(data)
        texts_idx = np.zeros((data_size, self.max_doc_length), dtype=np.long)

        for i, example in tqdm(enumerate(data), total=data_size, desc='transforming...'):
            exa_text_idx = np.array(example[0], dtype=np.long)
            cur_len = exa_text_idx.shape[0]
            if cur_len > self.max_doc_length:
                texts_idx[i] = exa_text_idx[:self.max_doc_length]
            else:
                texts_idx[i][:cur_len] = exa_text_idx

            exa_label_idx = np.array(list(map(lambda x: x-4, example[1])), dtype=np.long)
            labels_origin.append(exa_label_idx)

        labels_idx = MultiLabelBinarizer().fit_transform(labels_origin)
        batch, cur_label_size = labels_idx.shape
        if cur_label_size < self.label_size:
            labels_idx = np.concatenate([np.zeros((batch, self.label_size - cur_label_size), dtype=np.long),
                                         labels_idx], axis=1)
        return texts_idx, labels_idx
