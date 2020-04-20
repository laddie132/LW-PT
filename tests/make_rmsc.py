#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import os
import pickle
import json
from sklearn.model_selection import train_test_split


def spit_rmsc(data_path, save_path, random_seed=1):
    songs = os.listdir(data_path)
    songs.sort()  # make sure the same order for different machines

    # split the data
    songs_train, songs_test_valid = train_test_split(songs, test_size=0.3, random_state=random_seed)
    songs_test, songs_valid = train_test_split(songs_test_valid, test_size=0.3, random_state=random_seed)

    meta_data = {'songs_train': songs_train,
                 'songs_valid': songs_valid,
                 'songs_test': songs_test}
    transform_json(meta_data=meta_data, save_path=save_path, data_path=data_path)


def transform_json(save_path, data_path, meta_path=None, meta_data=None):
    if meta_path:
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


if __name__ == '__main__':
    print('splitting RMSC dataset...')
    spit_rmsc(data_path='data/rmsc/small',
              save_path='data/rmsc.data.v2')
    # transform_json(data_path='data/rmsc/small',
    #                meta_path='data/rmsc.pickle.meta',
    #                save_path='data/rmsc.data')
    print('finished.')
