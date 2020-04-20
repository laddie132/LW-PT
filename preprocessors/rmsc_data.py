#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import os
import json
import jieba
import logging
from tqdm import tqdm
from .base import BaseDataset

logger = logging.getLogger(__name__)


class RMSC(BaseDataset):
    """
    RMSC dataset
    """

    def __init__(self, data_path, random_seed):
        super(RMSC, self).__init__(h5_path='data/rmsc.h5',
                                   save_data_path='data/rmsc.pkl',
                                   save_meta_data_path='data/rmsc.pkl.meta',
                                   w2v_path='data/rmsc_word2vec.model',
                                   load_emb=False,
                                   emb_dim=100,
                                   max_vocab_size=None,
                                   max_sent_num=40,
                                   max_sent_len=20,
                                   max_doc_len=500,
                                   hier=True,
                                   random_seed=random_seed)

        self.data_path = 'data/rmsc' if data_path is '' else data_path

    def load_all_data(self):
        train_text_labels = self.load_data(os.path.join(self.data_path, 'rmsc.data.train.json'))
        val_text_labels = self.load_data(os.path.join(self.data_path, 'rmsc.data.valid.json'))
        test_text_labels = self.load_data(os.path.join(self.data_path, 'rmsc.data.test.json'))

        return train_text_labels, val_text_labels, test_text_labels

    def load_data(self, data_path):
        texts_labels = []
        with open(data_path, 'r') as f:
            data = json.load(f)
        for sample in tqdm(data, desc='loading data...'):
            cur_sents = []
            cur_labels = sample['tags']
            all_comments_dict = sample['all_short_comments'][:self.max_sent_num]

            for comment_dict in all_comments_dict:
                every_comment = comment_dict["comment"]
                every_comment_cut = "/".join(jieba.cut(every_comment, cut_all=False)).split('/')  # cur comment
                cur_sents.append(every_comment_cut)  # [[pl1],[pl2],...]

                self.texts_words_sum += len(every_comment_cut)
            self.texts_sents_sum += len(all_comments_dict)
            self.texts_labels_sum += len(cur_labels)

            self.all_texts.extend(cur_sents)
            self.all_labels.extend(cur_labels)

            texts_labels.append((cur_sents, cur_labels))
        return texts_labels
