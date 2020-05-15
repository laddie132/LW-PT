#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""LW-PT Module"""

import torch
import logging
from .base import BaseModule
from . import encoder
from .layers import get_embedding_layer

logger = logging.getLogger(__name__)


class LWPT(BaseModule):
    def __init__(self, config):
        super(LWPT, self).__init__(
            config,
            name='pt',
            model=LWPTTrainModel(config['model']))


class LWPTRep(BaseModule):
    def __init__(self, config):
        super(LWPTRep, self).__init__(
            config,
            name='pt',
            model=LWPTTestModel(config['model']))


class LWPTTrainModel(torch.nn.Module):
    """
    Documents representation model
    Args:
        model_config: config
    Inputs:
        tar_d: (batch, doc_sent_len, doc_word_len, emb_dim)
        cand_d: (batch, cand_doc_num, doc_sent_len, doc_word_len, emb_dim)
    Outputs:
        cand_d_prop: (batch, cand_doc_num)
    """

    def __init__(self, model_config):
        super(LWPTTrainModel, self).__init__()
        hierarchical = model_config['hierarchical']
        self.embedding_layer = get_embedding_layer(model_config)
        if hierarchical:
            self.tar_doc_encoder = encoder.HLWANEncoder(model_config)
            self.cand_doc_encoder = encoder.HLWANEncoder(model_config)

            # self.tar_doc_encoder = encoder.HANEncoder(model_config)
            # self.cand_doc_encoder = encoder.HANEncoder(model_config)
        else:
            self.tar_doc_encoder = encoder.LWBiRNNEncoder(model_config)
            self.cand_doc_encoder = encoder.LWBiRNNEncoder(model_config)

            # self.tar_doc_encoder = encoder.BiRNNEncoder(model_config)
            # self.cand_doc_encoder = encoder.BiRNNEncoder(model_config)

    def forward(self, tar_d, tar_mask, cand_ds, cand_mask, label):
        # embedding layer
        tar_d = self.embedding_layer(tar_d)
        cand_ds = self.embedding_layer(cand_ds)

        # target document encoder layer
        tar_doc_rep, _ = self.tar_doc_encoder(tar_d, tar_mask, label)

        # candidate documents encoder layer
        batch, cand_doc_num = cand_ds.size(0), cand_ds.size(1)
        new_size = [batch * cand_doc_num] + list(cand_ds.shape[2:])
        cand_docs_emb_flip = cand_ds.view(*new_size)

        new_size = [batch * cand_doc_num] + list(cand_mask.shape[2:])
        cand_docs_mask_flip = cand_mask.view(*new_size)

        cand_label = label.unsqueeze(1).expand(batch, cand_doc_num, -1).contiguous()
        cand_label = cand_label.view(batch * cand_doc_num, -1)

        cand_docs_rep_flip, _ = self.cand_doc_encoder(cand_docs_emb_flip, cand_docs_mask_flip, cand_label)
        cand_docs_rep = cand_docs_rep_flip.contiguous().view(batch, cand_doc_num, -1)

        # output layer
        cand_scores = torch.bmm(tar_doc_rep.unsqueeze(1),
                                cand_docs_rep.transpose(1, 2)).squeeze(1)  # (batch, cand_doc_num)
        cand_prob = torch.softmax(cand_scores, dim=-1)
        cand_logits = torch.log(cand_prob + 1e-8)     # to prevent Nan loss

        return cand_logits


class LWPTTestModel(torch.nn.Module):
    """
    Documents representation out model
    Args:
        model_config: config
    Inputs:
        doc: (batch, doc_sent_len, doc_word_len)
    Outputs:
        document_rep: (batch, label_size, hidden_size * 4)
    """

    def __init__(self, model_config):
        super(LWPTTestModel, self).__init__()
        hierarchical = model_config['hierarchical']
        self.embedding_layer = get_embedding_layer(model_config)
        if hierarchical:
            self.tar_doc_encoder = encoder.HLWANEncoder(model_config)
            self.cand_doc_encoder = encoder.HLWANEncoder(model_config)
        else:
            self.tar_doc_encoder = encoder.LWBiRNNEncoder(model_config)
            self.cand_doc_encoder = encoder.LWBiRNNEncoder(model_config)

    def forward(self, doc, doc_mask):
        # embedding layer
        doc = self.embedding_layer(doc)

        # doc encoder layer
        tar_doc_rep, _ = self.tar_doc_encoder(doc, doc_mask)
        cand_doc_rep, _ = self.cand_doc_encoder(doc, doc_mask)

        # doc representation: (batch, label_size, hidden_size * 4)
        doc_rep = torch.cat([tar_doc_rep, cand_doc_rep], dim=-1)

        return doc_rep
