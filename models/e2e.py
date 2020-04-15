#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""E2E models for multi-label text classification"""

import torch
from .layers import get_embedding_layer
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
        self.decoder = get_decoder(dec, hidden_size, label_size, qt=False, lw=False)

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
        self.decoder = get_decoder(dec, hidden_size, label_size, qt=False, lw=True)

    def forward(self, doc, *args):
        doc_emb = self.embedding_layer(doc)
        return self.decoder(self.encoder(doc_emb, *args)[0])


class LWQT(torch.nn.Module):
    def __init__(self, model_config):
        super(LWQT, self).__init__()
        hidden_size = model_config['hidden_size']
        label_size = model_config['label_size']
        hierarchical = model_config['hierarchical']
        self.fine_tune = model_config['fine_tune']
        dec = model_config['decoder']

        self.embedding_layer = get_embedding_layer(model_config)
        if hierarchical:
            self.tar_doc_encoder = encoder.HLWANEncoder(model_config)
            self.cand_doc_encoder = encoder.HLWANEncoder(model_config)
        else:
            self.tar_doc_encoder = encoder.LWBiRNNEncoder(model_config)
            self.cand_doc_encoder = encoder.LWBiRNNEncoder(model_config)

        self.decoder = get_decoder(dec, hidden_size, label_size, qt=True, lw=True)

    def forward(self, doc, *args):
        doc_emb = self.embedding_layer(doc)

        if self.fine_tune:
            tar_doc_rep = self.tar_doc_encoder(doc_emb, *args)[0]
            cand_doc_rep = self.cand_doc_encoder(doc_emb, *args)[0]
        else:
            with torch.no_grad():
                tar_doc_rep = self.tar_doc_encoder(doc_emb, *args)[0]
                cand_doc_rep = self.cand_doc_encoder(doc_emb, *args)[0]

        # doc representation: (batch, label_size, hidden_size * 4)
        doc_rep = torch.cat([tar_doc_rep, cand_doc_rep], dim=-1)
        return self.decoder(doc_rep)


def get_decoder(method, hidden_size, label_size, qt, lw):
    times = 4 if qt else 2
    if method == 'MLP':
        input_size = hidden_size * times * label_size if lw else hidden_size * times
        return decoder.LinearMLC(input_size, label_size)
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
