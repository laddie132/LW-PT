#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import pickle
import json
import torch
from sklearn.manifold import TSNE
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random
from gensim.models import KeyedVectors
# from whatlies import EmbeddingSet
# from whatlies.language import SpacyLanguage

sns.set()
mpl.rcParams['font.sans-serif'] = ['SimHei']


def transform_tsv(meta_data_path, doc_rep_path, save_path, dim=0):
    docs_rep = torch.load(doc_rep_path, map_location=torch.device('cpu'))
    test_doc_rep = docs_rep['test_doc_rep'][:, dim, :]
    test_doc_rep = test_doc_rep.numpy().tolist()

    with open(meta_data_path, 'rb') as f:
        meta_data = pickle.load(f)
    test_songs = meta_data['songs_test']

    with open(save_path, 'w') as f:
        for line in test_doc_rep:
            out = '\t'.join(list(map(lambda x: str(x), line))) + '\n'
            f.write(out)

    out_songs = list(map(lambda x: x + '\n', test_songs))
    with open(save_path + '.meta', 'w') as f:
        f.writelines(out_songs)


def transform_w2v(meta_data_path, doc_rep_path, save_path, dim=0):
    docs_rep = torch.load(doc_rep_path, map_location=torch.device('cpu'))
    test_doc_rep = docs_rep['test_doc_rep'][:, dim, :]
    size, emb = test_doc_rep.shape
    test_doc_rep = test_doc_rep.numpy().tolist()

    with open(meta_data_path, 'rb') as f:
        meta_data = pickle.load(f)
    test_songs = meta_data['songs_test']
    # print(meta_data['labels'])

    with open(save_path, 'w') as f:
        f.write('{} {}\n'.format(size, emb))
        for i, line in enumerate(test_doc_rep):
            name = test_songs[i].replace(' ', '_')
            out = name + ' ' + ' '.join(list(map(lambda x: str(x), line))) + '\n'
            f.write(out)


def load_test_json(test_path):
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    tag_d = {}
    for ele in test_data:
        tag_d[ele['name'].lower()] = ele['tags']
    return tag_d


def find_sim_songs(w2v_path, test_path, topn=10):
    qt_model = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
    songs_tag = load_test_json(test_path)

    example = 'Janes_Addiction_-_Nothings_Shocking'
    exp_tags = [
      "alternative",
      "rock"
    ]

    print(example, exp_tags)
    sim_names = qt_model.most_similar(example, topn=topn)

    tag_num = {}
    for ele in sim_names:
        name = ele[0].replace('_', ' ').lower()
        cur_tag = songs_tag[name]
        # print(name, ele[1], cur_tag, sep=', ')
        for t in cur_tag:
            tag_num[t] = tag_num.get(t, 0) + 1

    for k, v in tag_num.items():
        print('Tag: {} - {:.2%}'.format(k, v/topn))


def show_doc_emb(meta_data_path, doc_rep_path, save_path, nums, dim=0):
    docs_rep = torch.load(doc_rep_path, map_location=torch.device('cpu'))
    test_doc_rep = docs_rep['test_doc_rep'].numpy()

    with open(meta_data_path, 'rb') as f:
        meta_data = pickle.load(f)
    test_songs = meta_data['songs_test']

    assert len(test_songs) == test_doc_rep.shape[0]
    test_size, label_size, emb_size = test_doc_rep.shape

    tmp_idx = list(range(test_size))
    random.shuffle(tmp_idx)
    tmp_idx = tmp_idx[:nums]

    songs = [test_songs[i] for i in tmp_idx]
    vector = test_doc_rep[tmp_idx]
    # vector = vector.reshape((nums, -1))
    # vector = vector[:, dim, :]

    # visual
    plt.figure(figsize=(14, 10))

    for i in range(label_size):
        dim_vector = vector[:, i, :]

        # TSNE
        # tsne = TSNE(n_components=2, init='pca', verbose=1)
        # dim_emb = tsne.fit_transform(dim_vector)

        # SVD
        U, s, Vh = np.linalg.svd(dim_vector, full_matrices=False)
        dim_emb = U

        # visual
        plt.scatter(dim_emb[:, 0], dim_emb[:, 1], marker='^', label=str(i))

        # coord = dim_emb[:, 0:2]
        # plt.xlim((np.min(coord[:, 0]) - 0.1, np.max(coord[:, 0]) + 0.1))
        # plt.ylim((np.min(coord[:, 1]) - 0.1, np.max(coord[:, 1]) + 0.1))

    # for i in range(nums):
    #     x = embedd[i][0]
    #     y = embedd[i][1]
    #     plt.text(x, y, songs[i])
    plt.savefig(save_path)


# def show_whatlies():
#     words = ["cat", "dog", "fish", "kitten", "man", "woman",
#              "king", "queen", "doctor", "nurse"]
#
#     emb = EmbeddingSet(*[lang[w] for w in words])
#     emb.plot_interactive(x_axis=emb["man"], y_axis=emb["woman"])


if __name__ == '__main__':
    # transform_w2v(meta_data_path='data/rmsc.pkl.meta-label',
    #               doc_rep_path='data/rmsc_qt_rep.pt-cand3',
    #               save_path='data/rmsc_qt_rep.dim11.test.w2v',
    #               dim=11)
    find_sim_songs('data/rmsc_qt_rep.dim0.test.w2v',
                   test_path='data/rmsc/rmsc.data.test.json',
                   topn=50)
    # transform_tsv(meta_data_path='data/rmsc.pkl.meta',
    #               doc_rep_path='data/rmsc_qt_rep.pt-cand3',
    #               save_path='data/rmsc_qt_rep.test.tsv')
    # show_doc_emb(meta_data_path='data/rmsc.pkl.meta',
    #              doc_rep_path='data/rmsc_qt_rep.pt-cand3',
    #              save_path='data/rmsc_qt_rep.test.png',
    #              nums=50)
    # show_whatlies()
