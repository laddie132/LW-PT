#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import os
import logging
import nltk
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from .base import BaseDataset

logger = logging.getLogger(__name__)


class AAPD(BaseDataset):
    """
    AAPD dataset
    """

    def __init__(self, data_path, random_seed):
        super(AAPD, self).__init__(h5_path='data/aapd.h5',
                                   save_data_path='data/aapd.pickle',
                                   save_meta_data_path='data/aapd.pickle.meta',
                                   save_w2v_path='data/aapd_word2vec.model',
                                   emb_dim=100,
                                   max_vocab_size=None,
                                   random_seed=random_seed)
        self.max_sent = 40
        self.max_word = 20
        self.max_doc_length = 500

        self.data_path = 'data/aapd' if data_path is '' else data_path
        self.raw_texts_labels = {}

    def extract(self):
        logger.info('extracting dataset...')
        train_text_labels, train_texts, train_labels, train_word_sum, train_label_sum = \
            self.load_data(os.path.join(self.data_path, 'text_train'),
                           os.path.join(self.data_path, 'label_train'))
        val_text_labels, val_texts, val_labels, val_word_sum, val_label_sum = \
            self.load_data(os.path.join(self.data_path, 'text_val'),
                           os.path.join(self.data_path, 'label_val'))
        test_text_labels, test_texts, test_labels, test_word_sum, test_label_sum = \
            self.load_data(os.path.join(self.data_path, 'text_test'),
                           os.path.join(self.data_path, 'label_test'))

        self.raw_texts_labels['train'] = train_text_labels
        self.raw_texts_labels['valid'] = val_text_labels
        self.raw_texts_labels['test'] = test_text_labels

        total_texts = train_texts + val_texts + test_texts
        total_labels = train_labels.union(val_labels).union(test_labels)

        data_size = len(total_texts)
        self.attrs['data_size'] = data_size
        self.attrs['train_size'] = len(train_text_labels)
        self.attrs['valid_size'] = len(val_text_labels)
        self.attrs['test_size'] = len(test_text_labels)
        logger.info('Size: {}'.format(data_size))
        logger.info('Train size: {}'.format(len(train_text_labels)))
        logger.info('Valid size: {}'.format(len(val_text_labels)))
        logger.info('Test size: {}'.format(len(test_text_labels)))

        logger.info('Labels: {}'.format(len(total_labels)))
        self.attrs['label_size'] = len(total_labels)

        ave_text_len = (train_word_sum + val_word_sum + test_word_sum) * 1.0 / data_size
        ave_label_size = (train_label_sum + val_label_sum + test_label_sum) * 1.0 / data_size
        logger.info('Ave text len: {}'.format(ave_text_len))
        logger.info('Ave label size: {}'.format(ave_label_size))
        self.attrs['ave_text_len'] = ave_text_len
        self.attrs['ave_label_size'] = ave_label_size

        return total_texts, total_labels

    def load_data(self, text_path, label_path):
        texts_labels = []
        all_labels = []
        all_texts = []
        texts_word_sum = 0
        labels_sum = 0

        with open(text_path, 'r') as tf, open(label_path, 'r') as lf:
            for text, label in zip(tf, lf):
                if text != '' and label != '':
                    text = text.strip()
                    label = label.strip().split()
                    labels_sum += len(label)

                    text_split = nltk.word_tokenize(text)  # TODO: sentence-level split
                    all_texts.append(text_split)
                    texts_word_sum += len(text_split)

                    all_labels.extend(label)
                    texts_labels.append((text_split, label))
        all_labels = set(all_labels)

        return texts_labels, all_texts, all_labels, texts_word_sum, labels_sum

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
            exa_text_idx = np.array(list(map(self.word_index, example[0])), dtype=np.long)
            cur_len = exa_text_idx.shape[0]
            if cur_len > self.max_doc_length:
                texts_idx[i] = exa_text_idx[:self.max_doc_length]
            else:
                texts_idx[i][:cur_len] = exa_text_idx

            exa_label_idx = np.array(list(map(self.label_index, example[1])), dtype=np.long)
            labels_origin.append(exa_label_idx)

        labels_idx = MultiLabelBinarizer().fit_transform(labels_origin)
        return texts_idx, labels_idx
