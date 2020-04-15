#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import os
import json
import codecs
import jieba
import logging
import numpy as np
from tqdm import tqdm
from functools import reduce
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from .base import BaseDataset

logger = logging.getLogger(__name__)


# TODO: load songs from json data
class RMSC(BaseDataset):
    """
    RMSC dataset
    - modified from "https://github.com/lancopku/RMSC"
    """

    def __init__(self, data_path, random_seed):
        super(RMSC, self).__init__(h5_path='data/rmsc.h5',
                                   save_data_path='data/rmsc.pkl',
                                   save_meta_data_path='data/rmsc.pkl.meta',
                                   w2v_path='data/rmsc_word2vec.model',
                                   load_emb=False,
                                   emb_dim=100,
                                   max_vocab_size=None,
                                   random_seed=random_seed)
        self.max_sent = 40
        self.max_word = 20
        self.max_doc_length = 500

        self.data_path = 'data/rmsc/small' if data_path is '' else data_path
        self.songs = os.listdir(self.data_path)
        self.songs.sort()  # make sure the same order for different machines
        self.total_song_comments_and_tags = []

    def extract(self):
        total_tags = []
        total_docs = []

        for song in tqdm(self.songs, desc='extracting...'):
            with codecs.open(self.data_path + '/' + song, 'r+', encoding='utf-8') as f:
                a = f.read()
                dict_f = json.loads(a)
                every_song_comment_for_lstm_train = []
                all_comments_dict = dict_f['all_short_comments'][:self.max_sent]
                for comment_dict in all_comments_dict:
                    every_comment = comment_dict["comment"]
                    every_comment_cut = "/".join(jieba.cut(every_comment, cut_all=False)).split('/')  # cur comment
                    every_song_comment_for_lstm_train.append(every_comment_cut)  # [[pl1],[pl2],...]
                every_song_comments_and_tags = {"tags": dict_f['tags'], "comments": every_song_comment_for_lstm_train}
                total_docs.extend(every_song_comment_for_lstm_train)  # get all comments
                total_tags.extend(dict_f['tags'])  # get all classes
                self.total_song_comments_and_tags.append(every_song_comments_and_tags)  # all tags and comments
        total_tags = set(total_tags)

        logger.info('Songs: {}'.format(len(self.songs)))
        self.attrs['data_size'] = len(self.songs)

        logger.info('Tags: {}'.format(len(total_tags)))
        self.attrs['label_size'] = len(total_tags)

        return total_docs, total_tags

    def _transform_emb(self):
        labels_origin = []
        len_songs = len(self.total_song_comments_and_tags)
        embed_input = np.zeros((len_songs,
                                self.max_sent,
                                self.max_word,
                                self.emb_dim))

        # get input and label with embeddings
        len_every_comment_cutted = np.zeros((len_songs, self.max_sent))
        sum_comment_count = 0
        for i in tqdm(range(len_songs), desc='transforming...'):
            every_song_comment_words_embedding = np.zeros((self.max_sent, self.max_word, self.emb_dim))
            all_comments_in_every_song = self.total_song_comments_and_tags[i]["comments"]
            for j in range(len(all_comments_in_every_song)):
                every_comment_in_one_song = np.array(list(map(self.word_emb, all_comments_in_every_song[j])))
                every_comment_embedding = np.zeros((self.max_word, self.emb_dim))
                count_cur_comment = len(every_comment_in_one_song)
                sum_comment_count += count_cur_comment
                if count_cur_comment < self.max_word:
                    len_every_comment_cutted[i, j] = count_cur_comment
                    every_comment_embedding[0:count_cur_comment] = every_comment_in_one_song
                else:
                    every_comment_embedding = every_comment_in_one_song[0:self.max_word]
                    len_every_comment_cutted[i, j] = self.max_word
                    # logger.info(count_cur_comment)
                every_song_comment_words_embedding[j] = every_comment_embedding
            embed_input[i] = every_song_comment_words_embedding
            every_song_comment_tags_embedding = np.array(
                list(map(self.label_index, self.total_song_comments_and_tags[i]["tags"])))
            labels_origin.append(every_song_comment_tags_embedding)
        del self.total_song_comments_and_tags
        ave_sent_length = sum_comment_count / (len_songs * self.max_sent)
        logger.info("average comment length: {}".format(ave_sent_length))
        self.attrs['ave_sent_length'] = ave_sent_length
        labels = MultiLabelBinarizer().fit_transform(labels_origin)
        del labels_origin

        # split the data
        x_train, x_test_valid, y_train, y_test_valid, seq_train, seq_test_valid, songs_train, songs_test_valid = \
            train_test_split(embed_input, labels, len_every_comment_cutted, self.songs, test_size=0.3,
                             random_state=self.random_seed)
        # del soft_labels
        del embed_input
        del len_every_comment_cutted
        x_test, x_valid, y_test, y_valid, seq_test, seq_valid, songs_test, songs_valid = train_test_split(
            x_test_valid, y_test_valid, seq_test_valid, songs_test_valid, test_size=0.3, random_state=self.random_seed)
        # label_train, label_test_valid = train_test_split(labels, test_size=0.3, random_state=self.random_seed)
        # label_test, label_valid = train_test_split(label_test_valid, test_size=0.3, random_state=self.random_seed)
        logger.info('train: {}'.format(len(songs_train)))
        self.attrs['train_size'] = len(songs_train)
        logger.info('valid: {}'.format(len(songs_valid)))
        self.attrs['valid_size'] = len(songs_valid)
        logger.info('test: {}'.format(len(songs_test)))
        self.attrs['test_size'] = len(songs_test)

        data = {'x_train': x_train,
                'x_valid': x_valid,
                'x_test': x_test,
                'y_train': y_train,
                'y_valid': y_valid,
                'y_test': y_test,
                'seq_train': seq_train,
                'seq_valid': seq_valid,
                'seq_test': seq_test}
        meta_data = {'songs_train': songs_train,
                     'songs_valid': songs_valid,
                     'songs_test': songs_test,
                     'sorted_labels': self.sorted_labels}
        return data, meta_data

    def transform(self):
        labels_origin = []
        len_songs = len(self.total_song_comments_and_tags)
        comment = np.zeros((len_songs,
                            self.max_sent,
                            self.max_word), dtype=np.long)

        # get input and label with embeddings
        sum_comment_count = 0
        for i in tqdm(range(len_songs), desc='transforming...'):
            every_song_comment_words = np.zeros((self.max_sent, self.max_word), dtype=np.long)
            all_comments_in_every_song = self.total_song_comments_and_tags[i]["comments"]

            for j in range(len(all_comments_in_every_song)):
                every_comment_in_one_song = np.array(list(map(self.word_index, all_comments_in_every_song[j])),
                                                     dtype=np.long)
                every_comment = np.zeros((self.max_word,))

                count_cur_comment = len(every_comment_in_one_song)
                sum_comment_count += count_cur_comment

                if count_cur_comment < self.max_word:
                    every_comment[0:count_cur_comment] = every_comment_in_one_song
                else:
                    every_comment = every_comment_in_one_song[0:self.max_word]

                every_song_comment_words[j] = every_comment
            comment[i] = every_song_comment_words
            every_song_comment_tags_embedding = np.array(
                list(map(self.label_index, self.total_song_comments_and_tags[i]["tags"])))
            labels_origin.append(every_song_comment_tags_embedding)
        del self.total_song_comments_and_tags
        ave_sent_length = sum_comment_count / (len_songs * self.max_sent)

        logger.info("average comment length: {}".format(ave_sent_length))
        self.attrs['ave_sent_length'] = ave_sent_length
        labels = MultiLabelBinarizer().fit_transform(labels_origin)
        del labels_origin

        # split the data
        x_train, x_test_valid, y_train, y_test_valid, songs_train, songs_test_valid = \
            train_test_split(comment, labels, self.songs, test_size=0.3,
                             random_state=self.random_seed)
        del comment
        x_test, x_valid, y_test, y_valid, songs_test, songs_valid = train_test_split(
            x_test_valid, y_test_valid, songs_test_valid, test_size=0.3, random_state=self.random_seed)
        logger.info('train: {}'.format(len(songs_train)))
        self.attrs['train_size'] = len(songs_train)
        logger.info('valid: {}'.format(len(songs_valid)))
        self.attrs['valid_size'] = len(songs_valid)
        logger.info('test: {}'.format(len(songs_test)))
        self.attrs['test_size'] = len(songs_test)

        data = {'x_train': x_train,
                'x_valid': x_valid,
                'x_test': x_test,
                'y_train': y_train,
                'y_valid': y_valid,
                'y_test': y_test}
        meta_data = {'songs_train': songs_train,
                     'songs_valid': songs_valid,
                     'songs_test': songs_test}
        return data, meta_data

    def transform_non_hier(self):
        labels_origin = []
        len_songs = len(self.total_song_comments_and_tags)
        comment = np.zeros((len_songs,
                            self.max_doc_length), dtype=np.long)

        # get input and label with embeddings
        sum_comment_count = 0
        for i in tqdm(range(len_songs), desc='transforming...'):
            all_comments_in_every_song = self.total_song_comments_and_tags[i]["comments"]
            all_comments_in_every_song = list(reduce(lambda x, y: x + y, all_comments_in_every_song))
            ele = np.array(list(map(self.word_index, all_comments_in_every_song)), dtype=np.long)

            count_cur_comment = len(ele)
            sum_comment_count += count_cur_comment

            if count_cur_comment < self.max_doc_length:
                comment[i][0:count_cur_comment] = ele
            else:
                comment[i] = ele[0:self.max_doc_length]

            every_song_comment_tags_embedding = np.array(
                list(map(self.label_index, self.total_song_comments_and_tags[i]["tags"])))
            labels_origin.append(every_song_comment_tags_embedding)
        del self.total_song_comments_and_tags
        ave_sent_length = sum_comment_count / len_songs

        logger.info("average comment length: {}".format(ave_sent_length))
        self.attrs['ave_sent_length'] = ave_sent_length
        labels = MultiLabelBinarizer().fit_transform(labels_origin)
        del labels_origin

        # split the data
        x_train, x_test_valid, y_train, y_test_valid, songs_train, songs_test_valid = \
            train_test_split(comment, labels, self.songs, test_size=0.3,
                             random_state=self.random_seed)
        del comment
        x_test, x_valid, y_test, y_valid, songs_test, songs_valid = train_test_split(
            x_test_valid, y_test_valid, songs_test_valid, test_size=0.3, random_state=self.random_seed)
        logger.info('train: {}'.format(len(songs_train)))
        self.attrs['train_size'] = len(songs_train)
        logger.info('valid: {}'.format(len(songs_valid)))
        self.attrs['valid_size'] = len(songs_valid)
        logger.info('test: {}'.format(len(songs_test)))
        self.attrs['test_size'] = len(songs_test)

        data = {'x_train': x_train,
                'x_valid': x_valid,
                'x_test': x_test,
                'y_train': y_train,
                'y_valid': y_valid,
                'y_test': y_test}
        meta_data = {'songs_train': songs_train,
                     'songs_valid': songs_valid,
                     'songs_test': songs_test}
        return data, meta_data

    def transform_emb_ave(self):
        labels_origin = []
        len_songs = len(self.total_song_comments_and_tags)
        comment = np.zeros((len_songs, self.emb_dim), dtype=np.float)

        # get input and label with embeddings
        for i in tqdm(range(len_songs), desc='transforming...'):
            all_comments_in_every_song = self.total_song_comments_and_tags[i]["comments"]
            all_comments_in_every_song = list(reduce(lambda x, y: x + y, all_comments_in_every_song))
            ele = np.array(list(map(self.word_emb, all_comments_in_every_song)), dtype=np.float)
            comment[i] = ele.mean(axis=0)

            every_song_comment_tags_embedding = np.array(
                list(map(self.label_index, self.total_song_comments_and_tags[i]["tags"])))
            labels_origin.append(every_song_comment_tags_embedding)
        del self.total_song_comments_and_tags
        labels = MultiLabelBinarizer().fit_transform(labels_origin)
        del labels_origin

        # split the data
        x_train, x_test_valid, y_train, y_test_valid, songs_train, songs_test_valid = \
            train_test_split(comment, labels, self.songs, test_size=0.3,
                             random_state=self.random_seed)
        del comment
        x_test, x_valid, y_test, y_valid, songs_test, songs_valid = train_test_split(
            x_test_valid, y_test_valid, songs_test_valid, test_size=0.3, random_state=self.random_seed)
        logger.info('train: {}'.format(len(songs_train)))
        self.attrs['train_size'] = len(songs_train)
        logger.info('valid: {}'.format(len(songs_valid)))
        self.attrs['valid_size'] = len(songs_valid)
        logger.info('test: {}'.format(len(songs_test)))
        self.attrs['test_size'] = len(songs_test)

        data = {'x_train': x_train,
                'x_valid': x_valid,
                'x_test': x_test,
                'y_train': y_train,
                'y_valid': y_valid,
                'y_test': y_test}
        meta_data = {'songs_train': songs_train,
                     'songs_valid': songs_valid,
                     'songs_test': songs_test}
        return data, meta_data
