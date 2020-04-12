#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

from .layers import *


class LWBiGRUEncoder(torch.nn.Module):
    """
    Label-Wise Bidirectional GRU encoder on word-level for document representation
    Inputs:
        doc_emb: (batch, doc_len, emb_dim)
        doc_mask: (batch, doc_len)
        label: (batch, label_size)
    Outputs:
        doc_rep: (batch, hidden_size * 2)
    """
    def __init__(self, model_config):
        super(LWBiGRUEncoder, self).__init__()

        embedding_dim = model_config['embedding_dim']
        hidden_size = model_config['hidden_size']
        label_size = model_config['label_size']
        dropout_p = model_config['dropout_p']
        enable_layer_norm = model_config['layer_norm']

        self.doc_rnn = MyRNNBase(mode='GRU',
                                 input_size=embedding_dim,
                                 hidden_size=hidden_size,
                                 bidirectional=True,
                                 dropout_p=dropout_p,
                                 enable_layer_norm=enable_layer_norm,
                                 batch_first=True,
                                 num_layers=1)
        self.doc_attention = MultiHeadSelfAttention(in_features=hidden_size * 2,
                                                    labels=label_size)

    def forward(self, doc_emb, doc_mask, label):
        visual_parm = {}

        # (batch, doc_len, hidden_size * 2)
        doc_rep, _ = self.doc_rnn(doc_emb, doc_mask)

        # (batch, hidden_size * 2)
        doc_rep, doc_word_att_p = self.doc_attention(doc_rep, label, doc_mask)
        visual_parm['doc_word_att_p'] = doc_word_att_p

        return doc_rep, visual_parm


class HLWANEncoder(torch.nn.Module):
    """
    Hierarchical Label-Wise Attention Network for document representation
    Inputs:
        doc_emb: (batch, doc_sent_len, doc_word_len, emb_dim)
        doc_mask: (batch, doc_sent_len, doc_word_len)
        label: (batch, label_size)
    Outputs:
        doc_rep: (batch, hidden_size * 2)
    """

    def __init__(self, model_config):
        super(HLWANEncoder, self).__init__()

        embedding_dim = model_config['embedding_dim']
        hidden_size = model_config['hidden_size']
        label_size = model_config['label_size']
        dropout_p = model_config['dropout_p']
        enable_layer_norm = model_config['layer_norm']

        self.doc_word_rnn = MyRNNBase(mode='GRU',
                                      input_size=embedding_dim,
                                      hidden_size=hidden_size,
                                      bidirectional=True,
                                      dropout_p=dropout_p,
                                      enable_layer_norm=enable_layer_norm,
                                      batch_first=True,
                                      num_layers=1)
        self.doc_word_attention = MultiHeadSelfAttention(in_features=hidden_size * 2,
                                                         labels=label_size)

        self.doc_sentence_rnn = MyRNNBase(mode='GRU',
                                          input_size=hidden_size * 2,
                                          hidden_size=hidden_size,
                                          bidirectional=True,
                                          dropout_p=dropout_p,
                                          enable_layer_norm=enable_layer_norm,
                                          batch_first=True,
                                          num_layers=1)
        self.doc_sentence_attention = MultiHeadSelfAttention(in_features=hidden_size * 2,
                                                             labels=label_size)

    def forward(self, doc_emb, doc_mask, label):
        visual_parm = {}
        batch, doc_sent_len, doc_word_len, _ = doc_emb.size()

        doc_word_emb = doc_emb.view(batch * doc_sent_len, doc_word_len, -1)
        doc_word_mask = doc_mask.view(batch * doc_sent_len, doc_word_len)

        word_level_label = label.unsqueeze(1).expand(batch, doc_sent_len, -1).contiguous()
        word_level_label = word_level_label.view(batch * doc_sent_len, -1)

        # (batch * doc_sent_len, doc_word_len, hidden_size * 2)
        doc_word_rep, _ = self.doc_word_rnn(doc_word_emb, doc_word_mask)

        # (batch * doc_sent_len, hidden_size * 2)
        doc_sent_emb, doc_word_att_p = self.doc_word_attention(doc_word_rep, word_level_label, doc_word_mask)
        visual_parm['doc_word_att_p'] = doc_word_att_p

        # (batch, doc_sent_len, hidden_size * 2)
        doc_sent_emb = doc_sent_emb.view(batch, doc_sent_len, -1)
        doc_sent_mask = compute_top_layer_mask(doc_mask)

        # (batch, doc_sent_len, hidden_size * 2)
        doc_sent_rep, _ = self.doc_sentence_rnn(doc_sent_emb, doc_sent_mask)

        # (batch, hidden_size * 2)
        doc_rep, doc_sent_att_p = self.doc_sentence_attention(doc_sent_rep, label, doc_sent_mask)
        visual_parm['doc_sent_att_p'] = doc_sent_att_p

        return doc_rep, visual_parm


class BiGRUEncoder(torch.nn.Module):
    """
    Bidirectional GRU encoder on word-level for document representation
    Inputs:
        doc_emb: (batch, doc_len, emb_dim)
        doc_mask: (batch, doc_len)
        label: (batch, label_size)
    Outputs:
        doc_rep: (batch, hidden_size * 2)
    """
    def __init__(self, model_config):
        super(BiGRUEncoder, self).__init__()

        embedding_dim = model_config['embedding_dim']
        hidden_size = model_config['hidden_size']
        dropout_p = model_config['dropout_p']
        enable_layer_norm = model_config['layer_norm']

        self.doc_rnn = MyRNNBase(mode='GRU',
                                 input_size=embedding_dim,
                                 hidden_size=hidden_size,
                                 bidirectional=True,
                                 dropout_p=dropout_p,
                                 enable_layer_norm=enable_layer_norm,
                                 batch_first=True,
                                 num_layers=1)
        self.doc_attention = SelfAttention(in_features=hidden_size * 2)

    def forward(self, doc_emb, doc_mask):
        visual_parm = {}

        # (batch, doc_len, hidden_size * 2)
        doc_rep, _ = self.doc_word_rnn(doc_emb, doc_mask)

        # (batch, hidden_size * 2)
        doc_rep, doc_word_att_p = self.doc_attention(doc_rep, doc_mask)
        visual_parm['doc_word_att_p'] = doc_word_att_p

        return doc_rep, visual_parm


class HANEncoder(torch.nn.Module):
    """
    Hierarchical Attention Network for document representation
    Inputs:
        doc_emb: (batch, doc_sent_len, doc_word_len, emb_dim)
        doc_mask: (batch, doc_sent_len, doc_word_len)
    Outputs:
        doc_rep: (batch, hidden_size * 2)
    """

    def __init__(self, model_config):
        super(HANEncoder, self).__init__()

        embedding_dim = model_config['embedding_dim']
        hidden_size = model_config['hidden_size']
        dropout_p = model_config['dropout_p']
        enable_layer_norm = model_config['layer_norm']

        self.doc_word_rnn = MyRNNBase(mode='GRU',
                                      input_size=embedding_dim,
                                      hidden_size=hidden_size,
                                      bidirectional=True,
                                      dropout_p=dropout_p,
                                      enable_layer_norm=enable_layer_norm,
                                      batch_first=True,
                                      num_layers=1)
        self.doc_word_attention = SelfAttention(in_features=hidden_size * 2)

        self.doc_sentence_rnn = MyRNNBase(mode='GRU',
                                          input_size=hidden_size * 2,
                                          hidden_size=hidden_size,
                                          bidirectional=True,
                                          dropout_p=dropout_p,
                                          enable_layer_norm=enable_layer_norm,
                                          batch_first=True,
                                          num_layers=1)
        self.doc_sentence_attention = SelfAttention(in_features=hidden_size * 2)

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
