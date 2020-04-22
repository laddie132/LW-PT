#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import os
import logging
from .base import BaseDataset

logger = logging.getLogger(__name__)


class RCV1V2(BaseDataset):
    """
    RCV1-V2 dataset
    """

    def __init__(self, data_path, random_seed):
        super(RCV1V2, self).__init__(h5_path='data/rcv1v2.h5',
                                     save_data_path='data/rcv1v2.pkl',
                                     save_meta_data_path='data/rcv1v2.meta.json',
                                     w2v_path='data/rcv1v2_word2vec.model',
                                     load_emb=False,
                                     emb_dim=256,
                                     max_vocab_size=None,
                                     max_sent_num=15,
                                     max_sent_len=50,
                                     max_doc_len=500,
                                     hier=False,
                                     random_seed=random_seed)

        self.data_path = 'data/rcv1-v2/sgm' if data_path is '' else data_path

    def load_all_data(self):
        train_text_labels = self.load_data(os.path.join(self.data_path, 'train.src.id'),
                                           os.path.join(self.data_path, 'train.tgt.id'))
        val_text_labels = self.load_data(os.path.join(self.data_path, 'valid.src.id'),
                                         os.path.join(self.data_path, 'valid.tgt.id'))
        test_text_labels = self.load_data(os.path.join(self.data_path, 'test.src.id'),
                                          os.path.join(self.data_path, 'test.tgt.id'))

        return train_text_labels, val_text_labels, test_text_labels

    def load_data(self, text_path, label_path):
        texts_labels = []

        with open(text_path, 'r') as tf, open(label_path, 'r') as lf:
            for text, label in zip(tf, lf):
                if text != '' and label != '':
                    text = list(map(lambda x: int(x), text.strip().split()))
                    label = list(map(lambda x: int(x), label.strip().split()))
                    label = label[1:-1]

                    self.texts_labels_sum += len(label)
                    self.texts_words_sum += len(text)

                    self.all_texts.append(text)
                    self.all_labels.extend(label)

                    texts_labels.append((text, label))

        return texts_labels
