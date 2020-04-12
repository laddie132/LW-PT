#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""E2E models for multi-label text classification"""

import torch
from .layers import get_embedding_layer
from . import encoder, decoder


class BiGRU(torch.nn.Module):
    def __init__(self, model_config):
        super(BiGRU, self).__init__()
        hidden_size = model_config['hidden_size']
        label_size = model_config['label_size']

        self.embedding_layer = get_embedding_layer(model_config)
        self.encoder = encoder.BiGRUEncoder(model_config)

        input_size = hidden_size * 2
        self.decoder = decoder.LinearMLC(input_size, label_size)

    def forward(self, doc, *args):
        doc_emb = self.embedding_layer(doc)
        return self.decoder(self.encoder(doc_emb, *args)[0])


class LW_BiGRU(torch.nn.Module):
    def __init__(self, model_config):
        super(LW_BiGRU, self).__init__()
        hidden_size = model_config['hidden_size']
        label_size = model_config['label_size']
        self.embedding_layer = get_embedding_layer(model_config)
        self.encoder = encoder.LWBiGRUEncoder(model_config)

        input_size = hidden_size * 2
        self.decoder = decoder.LabelWiseMLC(input_size, label_size)

    def forward(self, doc, *args):
        doc_emb = self.embedding_layer(doc)
        return self.decoder(self.encoder(doc_emb, *args)[0])


class HAN(torch.nn.Module):
    def __init__(self, model_config):
        super(HAN, self).__init__()
        hidden_size = model_config['hidden_size']
        label_size = model_config['label_size']
        self.embedding_layer = get_embedding_layer(model_config)
        self.encoder = encoder.HANEncoder(model_config)

        input_size = hidden_size * 2
        self.decoder = decoder.LinearMLC(input_size, label_size)

    def forward(self, doc, *args):
        doc_emb = self.embedding_layer(doc)
        return self.decoder(self.encoder(doc_emb, *args)[0])


class HANLG(torch.nn.Module):
    def __init__(self, model_config):
        super(HANLG, self).__init__()
        hidden_size = model_config['hidden_size']
        label_size = model_config['label_size']
        self.embedding_layer = get_embedding_layer(model_config)
        self.encoder = encoder.HANEncoder(model_config)

        input_size = hidden_size * 2
        self.decoder = decoder.LabelGraphMLC(input_size, label_size)

    def forward(self, doc, *args):
        doc_emb = self.embedding_layer(doc)
        return self.decoder(self.encoder(doc_emb, *args)[0])


class HLWAN(torch.nn.Module):
    def __init__(self, model_config):
        super(HLWAN, self).__init__()
        hidden_size = model_config['hidden_size']
        label_size = model_config['label_size']
        self.embedding_layer = get_embedding_layer(model_config)
        self.tar_doc_encoder = encoder.HLWANEncoder(model_config)

        input_size = hidden_size * 2 * label_size
        self.decoder = decoder.LinearMLC(input_size, label_size)

    def forward(self, doc, *args):
        doc_emb = self.embedding_layer(doc)
        return self.decoder(self.tar_doc_encoder(doc_emb, *args)[0])


class HLWAN_QT(torch.nn.Module):
    def __init__(self, model_config):
        super(HLWAN_QT, self).__init__()
        hidden_size = model_config['hidden_size']
        label_size = model_config['label_size']
        self.embedding_layer = get_embedding_layer(model_config)
        self.tar_doc_encoder = encoder.HLWANEncoder(model_config)
        self.cand_doc_encoder = encoder.HLWANEncoder(model_config)

        input_size = hidden_size * 4 * label_size
        self.decoder = decoder.LinearMLC(input_size, label_size)

    def forward(self, doc, *args):
        doc_emb = self.embedding_layer(doc)
        with torch.no_grad():
            tar_doc_rep = self.tar_doc_encoder(doc_emb, *args)[0]
            cand_doc_rep = self.cand_doc_encoder(doc_emb, *args)[0]

            # doc representation: (batch, label_size, hidden_size * 4)
            doc_rep = torch.cat([tar_doc_rep, cand_doc_rep], dim=-1)

        return self.decoder(doc_rep)
