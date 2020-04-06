#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""DA-QT Module"""

import logging
import numpy as np
from .base import BaseModule
from .layers import *
from utils.functions import compute_mask, del_zeros_right, compute_top_layer_mask

logger = logging.getLogger(__name__)


class DAQT(BaseModule):
    def __init__(self, game_config):
        super(DAQT, self).__init__()
        self.game_config = game_config
        self.name = 'qt'

        self.in_checkpoint_path = game_config['checkpoint']['in_qt_checkpoint_path']
        self.in_weight_path = game_config['checkpoint']['in_qt_weight_path']
        self.out_checkpoint_path = game_config['checkpoint']['out_qt_checkpoint_path']
        self.out_weight_path = game_config['checkpoint']['out_qt_weight_path']

        self.model = DocRepQTTrainModel(game_config['model'])


class DocRepQTTrainModel(torch.nn.Module):
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
        super(DocRepQTTrainModel, self).__init__()

        self.model_config = model_config

        embedding_dim = model_config['embedding_dim']
        self.hidden_size = model_config['hidden_size']

        dropout_p = model_config['dropout_p']
        enable_layer_norm = model_config['layer_norm']

        self.tar_doc_encoder = DocRepQTEncoder(embedding_dim, self.hidden_size, dropout_p, enable_layer_norm)
        self.cand_doc_encoder = DocRepQTEncoder(embedding_dim, self.hidden_size, dropout_p, enable_layer_norm)

    def forward(self, tar_d, tar_mask, cand_ds, cand_mask, label):
        # target document encoder layer
        tar_doc_rep, _ = self.tar_doc_encoder(tar_d, tar_mask)

        # candidate documents encoder layer
        batch, cand_doc_num = cand_ds.size(0), cand_ds.size(1)
        new_size = [batch * cand_doc_num] + list(cand_ds.shape[2:])
        cand_docs_emb_flip = cand_ds.view(*new_size)

        new_size = [batch * cand_doc_num] + list(cand_ds.shape[2:])
        cand_docs_mask_flip = cand_ds.view(*new_size)

        cand_docs_rep_flip, _ = self.cand_doc_encoder(cand_docs_emb_flip, cand_docs_mask_flip)
        cand_docs_rep = cand_docs_rep_flip.contiguous().view(batch, cand_doc_num, -1)

        # output layer
        cand_scores = torch.bmm(tar_doc_rep.unsqueeze(1),
                                cand_docs_rep.transpose(1, 2)).squeeze(1)  # (batch, cand_doc_num)
        cand_logits = torch.log_softmax(cand_scores, dim=-1)

        return cand_logits


class DocRepQTTestModel(torch.nn.Module):
    """
    Documents representation out model
    Args:
        model_config: config
    Inputs:
        doc: (batch, doc_sent_len, doc_word_len)
    Outputs:
        document_rep: (batch, hidden_size * 4)
    """

    def __init__(self, model_config):
        super(DocRepQTTestModel, self).__init__()

        self.model_config = model_config

        embedding_dim = model_config['embedding_dim']

        self.hidden_size = model_config['hidden_size']

        dropout_p = model_config['dropout_p']
        enable_layer_norm = model_config['layer_norm']

        self.tar_doc_encoder = DocRepQTEncoder(embedding_dim, self.hidden_size, dropout_p, enable_layer_norm)
        self.cand_doc_encoder = DocRepQTEncoder(embedding_dim, self.hidden_size, dropout_p, enable_layer_norm)

    def forward(self, doc):
        doc, _ = del_zeros_right(doc)
        _, sent_right_idx = del_zeros_right(doc.sum(-1))
        doc = doc[:, :sent_right_idx, :]

        # embedding layer
        doc_emb = self.embedding_layer(doc)
        doc_mask = compute_mask(doc)

        # doc encoder layer
        tar_doc_rep, _ = self.tar_doc_encoder(doc_emb, doc_mask)
        cand_doc_rep, _ = self.cand_doc_encoder(doc_emb, doc_mask)

        # doc representation
        doc_rep = torch.cat([tar_doc_rep, cand_doc_rep], dim=-1)

        return doc_rep


class DocRepQTEncoder(torch.nn.Module):
    """
    Documents representation model
    Inputs:
        doc_emb: (batch, doc_sent_len, doc_word_len, emb_dim)
        doc_mask: (batch, doc_sent_len, doc_word_len)
    Outputs:
        doc_rep: (batch, hidden_size * 2)
    """

    def __init__(self, embedding_dim, hidden_size, dropout_p, enable_layer_norm):
        super(DocRepQTEncoder, self).__init__()

        self.hidden_size = hidden_size

        self.dropout_layer = torch.nn.Dropout(p=dropout_p)
        self.doc_word_rnn = MyRNNBase(mode='GRU',
                                      input_size=embedding_dim,
                                      hidden_size=self.hidden_size,
                                      bidirectional=True,
                                      dropout_p=dropout_p,
                                      enable_layer_norm=enable_layer_norm,
                                      batch_first=True,
                                      num_layers=1)
        self.doc_word_attention = SelfAttention(hidden_size=self.hidden_size * 2)

        self.doc_sentence_rnn = MyRNNBase(mode='GRU',
                                          input_size=self.hidden_size * 2,
                                          hidden_size=self.hidden_size,
                                          bidirectional=True,
                                          dropout_p=dropout_p,
                                          enable_layer_norm=enable_layer_norm,
                                          batch_first=True,
                                          num_layers=1)
        self.doc_sentence_attention = SelfAttention(hidden_size=self.hidden_size * 2)

    def forward(self, doc_emb, doc_mask):
        visual_parm = {}
        batch, doc_sent_len, doc_word_len, _ = doc_emb.size()

        doc_word_emb = doc_emb.view(batch * doc_sent_len, doc_word_len, -1)
        doc_word_mask = doc_mask.view(batch * doc_sent_len, doc_word_len)

        # (batch * doc_sent_len, doc_word_len, hidden_size * 2)
        doc_word_rep, _ = self.doc_word_rnn(doc_word_emb, doc_word_mask)

        # (batch * doc_sent_len, hidden_size * 2)
        doc_sent_emb, doc_word_att_p = self.doc_word_attention(doc_word_rep, doc_word_mask)
        visual_parm['doc_word_att_p'] = doc_word_att_p

        # (batch, doc_sent_len, hidden_size * 2)
        doc_sent_emb = doc_sent_emb.view(batch, doc_sent_len, -1)
        doc_sent_mask = compute_top_layer_mask(doc_mask)

        # (batch, doc_sent_len, hidden_size * 2)
        doc_sent_rep, _ = self.doc_sentence_rnn(doc_sent_emb, doc_sent_mask)

        # (batch, hidden_size * 2)
        doc_rep, doc_sent_att_p = self.doc_sentence_attention(doc_sent_rep, doc_sent_mask)
        visual_parm['doc_sent_att_p'] = doc_sent_att_p

        return doc_rep, visual_parm
