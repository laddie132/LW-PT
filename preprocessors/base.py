#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import h5py
import pickle
import logging
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models.word2vec import Word2Vec

logger = logging.getLogger(__name__)


class BaseDataset:
    _compress_option = dict(compression="gzip", compression_opts=9, shuffle=False)

    def __init__(self, h5_path, save_data_path, save_meta_data_path, w2v_path, load_emb,
                 emb_dim, max_vocab_size, max_sent_num, max_sent_len, max_doc_len, hier, random_seed):
        self.random_seed = random_seed
        self.emb_dim = emb_dim
        self.load_emb = load_emb
        self.max_vocab_size = max_vocab_size

        self.max_sent_num = max_sent_num
        self.max_sent_len = max_sent_len
        self.max_doc_len = max_doc_len
        self.hier = hier

        # path
        self.h5_path = h5_path
        self.save_data_path = save_data_path
        self.save_meta_data_path = save_meta_data_path
        self.w2v_path = w2v_path

        # data
        self.word2vec = None
        self.attrs = {}  # attributes

        self.dict_label = {}  # tags dictionary
        self.sorted_labels = []

        # raw data
        self.raw_texts_labels = {}

        self.all_labels = []
        self.all_texts = []
        self.texts_words_sum = 0
        self.texts_sents_sum = 0
        self.texts_labels_sum = 0

    def build(self):
        """
        ETL pipeline
        :return:
        """
        logger.info('building {} dataset...'.format(self.__class__.__name__))
        self.extract()
        self.train_emb()

        data, meta_data = self.transform()
        meta_data['labels'] = self.sorted_labels
        self.save(data, meta_data)
        # self.save_h5(data, meta_data)

    def load_all_data(self):
        return NotImplementedError

    def extract(self):
        logger.info('extracting dataset...')
        train_text_labels, val_text_labels, test_text_labels = self.load_all_data()

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

        self.all_labels = list(set(self.all_labels))
        logger.info('Labels: {}'.format(len(self.all_labels)))
        self.attrs['label_size'] = len(self.all_labels)

        ave_text_len = self.texts_words_sum * 1.0 / data_size
        ave_label_size = self.texts_labels_sum * 1.0 / data_size
        ave_sent_num = self.texts_sents_sum * 1.0 / data_size
        logger.info('Ave text len: {}'.format(ave_text_len))
        logger.info('Ave sent num: {}'.format(ave_sent_num))
        logger.info('Ave label size: {}'.format(ave_label_size))
        self.attrs['ave_text_len'] = ave_text_len
        self.attrs['ave_sent_num'] = ave_sent_num
        self.attrs['ave_label_size'] = ave_label_size

    def transform(self):
        if not self.hier:
            x_train, y_train = self.t5_data(self.raw_texts_labels['train'])
            x_valid, y_valid = self.t5_data(self.raw_texts_labels['valid'])
            x_test, y_test = self.t5_data(self.raw_texts_labels['test'])
        else:
            x_train, y_train = self.t5_data_hier(self.raw_texts_labels['train'])
            x_valid, y_valid = self.t5_data_hier(self.raw_texts_labels['valid'])
            x_test, y_test = self.t5_data_hier(self.raw_texts_labels['test'])

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
        texts_idx = np.zeros((data_size, self.max_doc_len), dtype=np.long)

        for i, example in tqdm(enumerate(data), total=data_size, desc='transforming...'):
            exa_text_idx = np.array(list(map(self.word_index, example[0])), dtype=np.long)
            cur_len = exa_text_idx.shape[0]
            if cur_len > self.max_doc_len:
                texts_idx[i] = exa_text_idx[:self.max_doc_len]
            else:
                texts_idx[i][:cur_len] = exa_text_idx

            exa_label_idx = np.array(list(map(self.label_index, example[1])), dtype=np.long)
            labels_origin.append(exa_label_idx)

        labels_idx = MultiLabelBinarizer().fit_transform(labels_origin)
        batch, cur_label_size = labels_idx.shape
        if cur_label_size < len(self.sorted_labels):
            labels_idx = np.concatenate([np.zeros((batch, len(self.sorted_labels) - cur_label_size), dtype=np.long),
                                         labels_idx], axis=1)
        return texts_idx, labels_idx

    def t5_data_hier(self, data):
        labels_origin = []
        data_size = len(data)
        texts_idx = np.zeros((data_size, self.max_sent_num, self.max_sent_len), dtype=np.long)

        for i, example in tqdm(enumerate(data), total=data_size, desc='transforming...'):
            cur_text = np.zeros((self.max_sent_num, self.max_sent_len), dtype=np.long)
            for j, sent in enumerate(example[0]):
                if j >= self.max_sent_num:
                    break

                sent_idx = np.array(list(map(self.word_index, sent)), dtype=np.long)
                cur_len = sent_idx.shape[0]
                if cur_len > self.max_sent_len:
                    cur_text[j] = sent_idx[:self.max_sent_len]
                else:
                    cur_text[j][:cur_len] = sent_idx

            texts_idx[i] = cur_text
            exa_label_idx = np.array(list(map(self.label_index, example[1])), dtype=np.long)
            labels_origin.append(exa_label_idx)

        labels_idx = MultiLabelBinarizer().fit_transform(labels_origin)
        batch, cur_label_size = labels_idx.shape
        if cur_label_size < len(self.sorted_labels):
            labels_idx = np.concatenate([np.zeros((batch, len(self.sorted_labels) - cur_label_size), dtype=np.long),
                                         labels_idx], axis=1)
        return texts_idx, labels_idx

    def train_emb(self):
        if self.load_emb:
            logger.info('loading word2vec...')
            self.word2vec = Word2Vec.load(self.w2v_path)  # load pre-trained embeddings
        else:
            logger.info('training word2vec...')
            self.word2vec = Word2Vec(self.all_texts, size=self.emb_dim, max_vocab_size=self.max_vocab_size,
                                     workers=1, min_count=1, seed=self.random_seed)
            self.word2vec.save(self.w2v_path, sep_limit=0, separately=['vectors'],
                               ignore=['vectors_lockf', 'syn1neg', 'cum_table'])
            self.word2vec.wv.save_word2vec_format(self.w2v_path + '.txt')
            logger.info("word2vec info: {}".format(self.word2vec))

        # labels dictionary
        self.sorted_labels = sorted(list(self.all_labels))
        for i, t in enumerate(self.sorted_labels):
            self.dict_label[t] = i

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
        logger.info('saving data...')
        with open(self.save_data_path, 'wb') as f:
            pickle.dump(data, f)

        logger.info('saving meta-data...')
        with open(self.save_meta_data_path, 'wb') as f:
            pickle.dump({**meta_data, **self.attrs}, f)

    def save_h5(self, data, meta_data):
        logger.info('saving hdf5 data...')
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
