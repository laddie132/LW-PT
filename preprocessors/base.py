#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import h5py
import pickle
import numpy as np
from gensim.models.word2vec import Word2Vec


class BaseDataset:
    _compress_option = dict(compression="gzip", compression_opts=9, shuffle=False)

    def __init__(self, h5_path, save_data_path, save_meta_data_path, save_w2v_path,
                 emb_dim, max_vocab_size, random_seed):
        self.random_seed = random_seed
        self.emb_dim = emb_dim
        self.max_vocab_size = max_vocab_size

        # path
        self.h5_path = h5_path
        self.save_data_path = save_data_path
        self.save_meta_data_path = save_meta_data_path
        self.save_w2v_path = save_w2v_path

        # data
        self.word2vec = None
        self.attrs = {}  # attributes
        self.dict_label = {}  # tags dictionary
        self.sorted_labels = []

    def build(self):
        """
        ETL pipeline
        :return:
        """
        print('building {} dataset...'.format(self.__class__.__name__))
        total_docs, total_labels = self.extract()
        self.train_emb(total_docs, total_labels)

        data, meta_data = self.transform()
        meta_data['labels'] = self.sorted_labels
        self.save(data, meta_data)
        # self.save_h5(data, meta_data)

    def extract(self):
        return NotImplementedError

    def transform(self):
        return NotImplementedError

    def train_emb(self, total_docs, total_labels):
        print('training word2vec...')
        self.word2vec = Word2Vec(total_docs, size=self.emb_dim, max_vocab_size=self.max_vocab_size,
                                 workers=1, min_count=1, seed=self.random_seed)
        # self.word2vec = Word2Vec.load('data/rmsc_word2vec.model')     # load pre-trained embeddings
        self.word2vec.save(self.save_w2v_path)
        self.word2vec.wv.save_word2vec_format(self.save_w2v_path + '.txt')
        print("word2vec info:", self.word2vec)

        # labels dictionary
        self.sorted_labels = sorted(list(total_labels))
        for i, t in enumerate(self.sorted_labels):  # TODO: check the dict_tags
            self.dict_label[t] = i + 1

    def word_emb(self, word):
        return self.word2vec[word]

    def word_index(self, word):
        return self.word2vec.wv.vocab[word].index + 1  # padding index on zero

    def label_index(self, tag):
        return self.dict_label[tag]

    def save(self, data, meta_data):
        """
        Save to file
        :return:
        """
        print('saving data...')
        with open(self.save_data_path, 'wb') as f:
            pickle.dump(data, f)

        print('saving meta-data...')
        with open(self.save_meta_data_path, 'wb') as f:
            pickle.dump({**meta_data, **self.attrs}, f)

    def save_h5(self, data, meta_data):
        print('saving hdf5 data...')
        f = h5py.File(self.h5_path, 'w')
        str_dt = h5py.special_dtype(vlen=str)

        # attributes
        for attr_name in self.attrs:
            f.attrs[attr_name] = self.attrs[attr_name]

        # meta_data
        f_meta_data = f.create_group('meta_data')
        for name, value in meta_data.items():
            value = np.array(value, dtype=np.str)
            cur_meta_data = f_meta_data.create_dataset(name, value.shape, dtype=str_dt, **self._compress_option)
            cur_meta_data[...] = value

        # data
        f_data = f.create_group('data')
        for name, value in data.items():
            cur_data = f_data.create_dataset(name, value.shape, dtype=value.dtype, **self._compress_option)
            cur_data[...] = value

        f.flush()
        f.close()
