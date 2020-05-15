#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""E2E models for multi-label text classification"""

import torch
from .layers import get_embedding_layer, SFU
from . import encoder, decoder


class CNN(torch.nn.Module):
    def __init__(self, model_config):
        super(CNN, self).__init__()
        label_size = model_config['label_size']
        self.embedding_layer = get_embedding_layer(model_config)
        self.encoder = encoder.CNNEncoder(model_config)

        input_size = 300
        self.decoder = decoder.LinearMLC(input_size, label_size)

    def forward(self, doc, *args):
        doc_emb = self.embedding_layer(doc)
        return self.decoder(self.encoder(doc_emb, *args)[0])


class RNN(torch.nn.Module):
    def __init__(self, model_config):
        super(RNN, self).__init__()
        hidden_size = model_config['hidden_size']
        label_size = model_config['label_size']
        hierarchical = model_config['hierarchical']
        dec = model_config['decoder']

        self.embedding_layer = get_embedding_layer(model_config)
        if hierarchical:
            self.encoder = encoder.HANEncoder(model_config)
        else:
            self.encoder = encoder.BiRNNEncoder(model_config)
        self.decoder = get_decoder(dec, hidden_size, label_size, pt=False, lw=False)

    def forward(self, doc, *args):
        doc_emb = self.embedding_layer(doc)
        return self.decoder(self.encoder(doc_emb, *args)[0])


class LWAN(torch.nn.Module):
    def __init__(self, model_config):
        super(LWAN, self).__init__()
        hidden_size = model_config['hidden_size']
        label_size = model_config['label_size']
        hierarchical = model_config['hierarchical']
        dec = model_config['decoder']

        self.embedding_layer = get_embedding_layer(model_config)
        if hierarchical:
            self.encoder = encoder.HLWANEncoder(model_config)
        else:
            self.encoder = encoder.LWBiRNNEncoder(model_config)
        self.decoder = get_decoder(dec, hidden_size, label_size, pt=False, lw=True)

    def forward(self, doc, *args):
        doc_emb = self.embedding_layer(doc)
        return self.decoder(self.encoder(doc_emb, *args)[0])


class LWPT(torch.nn.Module):
    def __init__(self, model_config):
        super(LWPT, self).__init__()
        hidden_size = model_config['hidden_size']
        label_size = model_config['label_size']
        hierarchical = model_config['hierarchical']
        self.fine_tune = model_config['fine_tune']
        dec = model_config['decoder']
        self.flag = False

        self.embedding_layer = get_embedding_layer(model_config)
        if hierarchical:
            self.tar_doc_encoder = encoder.HLWANEncoder(model_config)
            self.cand_doc_encoder = encoder.HLWANEncoder(model_config)
        else:
            self.tar_doc_encoder = encoder.LWBiRNNEncoder(model_config)
            self.cand_doc_encoder = encoder.LWBiRNNEncoder(model_config)

        self.decoder = get_decoder(dec, hidden_size, label_size, pt=True, lw=True)
        # self.sfu = SFU(hidden_size * 2, hidden_size * 2)

    def forward(self, doc, *args):
        self.flag = not self.flag
        doc_emb = self.embedding_layer(doc)

        if self.fine_tune:
            tar_doc_rep = self.tar_doc_encoder(doc_emb, *args)[0]
            cand_doc_rep = self.cand_doc_encoder(doc_emb, *args)[0]
            # if self.flag:
            #     tar_doc_rep = self.tar_doc_encoder(doc_emb, *args)[0]
            #     with torch.no_grad():
            #         cand_doc_rep = self.cand_doc_encoder(doc_emb, *args)[0]
            # else:
            #     with torch.no_grad():
            #         tar_doc_rep = self.tar_doc_encoder(doc_emb, *args)[0]
            #     cand_doc_rep = self.cand_doc_encoder(doc_emb, *args)[0]
        else:
            # self.tar_doc_encoder.eval()
            # self.cand_doc_encoder.eval()
            with torch.no_grad():
                tar_doc_rep = self.tar_doc_encoder(doc_emb, *args)[0]
                cand_doc_rep = self.cand_doc_encoder(doc_emb, *args)[0]

        # doc representation: (batch, label_size, hidden_size * 4)
        doc_rep = torch.cat([tar_doc_rep, cand_doc_rep], dim=-1)
        # doc_rep = (tar_doc_rep + cand_doc_rep) / 2
        # doc_rep = self.sfu(tar_doc_rep, cand_doc_rep)

        return self.decoder(doc_rep)


def get_decoder(method, hidden_size, label_size, pt, lw):
    times = 4 if pt else 2
    if method == 'Linear':
        input_size = hidden_size * times * label_size if lw else hidden_size * times
        return decoder.LinearMLC(input_size, label_size)
    elif method == 'MLP':
        input_size = hidden_size * times * label_size if lw else hidden_size * times
        return decoder.TwoLinearMLC(input_size, hidden_size * 10, label_size)
    elif method == 'LW':
        input_size = hidden_size * times
        return decoder.LabelWiseMLC(input_size, label_size)
    elif method == 'LG':
        input_size = hidden_size * times * label_size if lw else hidden_size * times
        return decoder.LabelGraphMLC(input_size, label_size)
    elif method == 'LWLG':
        input_size = hidden_size * times
        return decoder.LabelWiseGraphMLC(input_size, label_size)
    raise ValueError("Invalid decoder method: " + method)
