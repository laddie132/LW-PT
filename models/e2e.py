#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import torch
from .layers import get_embedding_layer
from . import encoder, decoder


class BiGRU(torch.nn.Module):
    def __init__(self, model_config):
        super(BiGRU, self).__init__()
        self.embedding_layer = get_embedding_layer(model_config)
        self.encoder = encoder.BiGRUEncoder(model_config)
        self.decoder = decoder.LinearMLC(model_config, qt=False)

    def forward(self, doc, *args):
        doc_emb = self.embedding_layer(doc)
        return self.decoder(self.encoder(doc_emb, *args)[0])


# class LW_BiGRU(torch.nn.Module):
#     def __init__(self, model_config):
#         super(LW_BiGRU, self).__init__()
#         self.embedding_layer = get_embedding_layer(model_config)
#         self.encoder = encoder.LWBiGRUEncoder(model_config)
#         self.decoder = decoder.LabelWiseMLC(model_config, qt=False)
#
#     def forward(self, doc, *args):
#         doc_emb = self.embedding_layer(doc)
#         return self.decoder(self.encoder(doc_emb, *args)[0])


class HAN(torch.nn.Module):
    def __init__(self, model_config):
        super(HAN, self).__init__()
        self.embedding_layer = get_embedding_layer(model_config)
        self.encoder = encoder.HANEncoder(model_config)
        self.decoder = decoder.LinearMLC(model_config, qt=False)

    def forward(self, doc, *args):
        doc_emb = self.embedding_layer(doc)
        return self.decoder(self.encoder(doc_emb, *args)[0])