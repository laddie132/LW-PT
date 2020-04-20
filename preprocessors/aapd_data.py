#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import os
import logging
import nltk
import re
from .base import BaseDataset

logger = logging.getLogger(__name__)


class AAPD(BaseDataset):
    """
    AAPD dataset
    """

    def __init__(self, data_path, random_seed):
        super(AAPD, self).__init__(h5_path='data/aapd.h5',
                                   save_data_path='data/aapd.pkl',
                                   save_meta_data_path='data/aapd.pkl.meta',
                                   w2v_path='data/aapd_word2vec.model',
                                   load_emb=False,
                                   emb_dim=256,
                                   max_vocab_size=None,
                                   max_sent_num=15,
                                   max_sent_len=50,
                                   max_doc_len=500,
                                   hier=False,
                                   random_seed=random_seed)
        self.data_path = 'data/aapd' if data_path is '' else data_path

    def load_all_data(self):
        train_text_labels = self.load_data(os.path.join(self.data_path, 'text_train'),
                                           os.path.join(self.data_path, 'label_train'))
        val_text_labels = self.load_data(os.path.join(self.data_path, 'text_val'),
                                         os.path.join(self.data_path, 'label_val'))
        test_text_labels = self.load_data(os.path.join(self.data_path, 'text_test'),
                                          os.path.join(self.data_path, 'label_test'))
        return train_text_labels, val_text_labels, test_text_labels

    def load_data(self, text_path, label_path):
        texts_labels = []

        with open(text_path, 'r') as tf, open(label_path, 'r') as lf:
            for text, label in zip(tf, lf):
                if text != '' and label != '':
                    text = text.strip()
                    label = label.strip().split()
                    self.texts_labels_sum += len(label)

                    text_split = nltk.word_tokenize(text)
                    self.all_texts.append(text_split)
                    self.texts_words_sum += len(text_split)

                    self.all_labels.extend(label)
                    texts_labels.append((text_split, label))

        return texts_labels


class AAPDHier(BaseDataset):
    """
    AAPD dataset
    """

    def __init__(self, data_path, random_seed):
        super(AAPDHier, self).__init__(h5_path='data/aapd_hier.h5',
                                       save_data_path='data/aapd_hier.pkl',
                                       save_meta_data_path='data/aapd_hier.pkl.meta',
                                       w2v_path='data/aapd_word2vec.model',
                                       load_emb=True,
                                       emb_dim=256,
                                       max_vocab_size=None,
                                       max_sent_num=15,
                                       max_sent_len=50,
                                       max_doc_len=500,
                                       hier=True,
                                       random_seed=random_seed)
        self.data_path = 'data/aapd' if data_path is '' else data_path

    def load_all_data(self):
        train_text_labels = self.load_data(os.path.join(self.data_path, 'text_train'),
                                           os.path.join(self.data_path, 'label_train'))
        val_text_labels = self.load_data(os.path.join(self.data_path, 'text_val'),
                                         os.path.join(self.data_path, 'label_val'))
        test_text_labels = self.load_data(os.path.join(self.data_path, 'text_test'),
                                          os.path.join(self.data_path, 'label_test'))
        return train_text_labels, val_text_labels, test_text_labels

    def load_data(self, text_path, label_path):
        texts_labels = []

        with open(text_path, 'r') as tf, open(label_path, 'r') as lf:
            for text, label in zip(tf, lf):
                if text != '' and label != '':
                    label = label.strip().split()
                    self.texts_labels_sum += len(label)

                    cur_sents = []
                    for sent in re.split('[，,.。？?]', text.strip()):
                        sent_split = nltk.word_tokenize(sent)
                        cur_sents.append(sent_split)
                        self.all_texts.append(sent_split)
                        self.texts_words_sum += len(sent_split)

                    self.texts_sents_sum += len(cur_sents)
                    self.all_labels.extend(label)
                    texts_labels.append((cur_sents, label))
        return texts_labels
