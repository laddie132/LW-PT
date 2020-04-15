#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import os
import pickle
import json


def transform_json(meta_path, save_path, data_path='data/rmsc/small'):
    with open(meta_path, 'rb') as f:
        meta_data = pickle.load(f)

    def _load_data(songs):
        all_songs = []
        for s in songs:
            cur_path = os.path.join(data_path, s)
            with open(cur_path, 'r') as f:
                a = f.read()
                dict_f = json.loads(a)
                all_songs.append(dict_f)

        return all_songs

    train_data = _load_data(meta_data['songs_train'])
    valid_data = _load_data(meta_data['songs_valid'])
    test_data = _load_data(meta_data['songs_test'])

    with open(save_path + '.train.json', 'w') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(save_path + '.valid.json', 'w') as f:
        json.dump(valid_data, f, ensure_ascii=False, indent=2)

    with open(save_path + '.test.json', 'w') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)


transform_json(meta_path='data/rmsc.pickle.meta',
               save_path='data/rmsc.data')
