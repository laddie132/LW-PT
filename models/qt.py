#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""DA-QT Module"""

import numpy as np
import logging
from .base import BaseModule
from .layers import *
from datareaders.vocabulary import Vocabulary
from utils.functions import compute_top_layer_mask

logger = logging.getLogger(__name__)


class DAQT(BaseModule):
    def __init__(self, config):
        super(DAQT, self).__init__()
        self.name = 'qt'

        self.in_checkpoint_path = config['checkpoint']['in_qt_checkpoint_path']
        self.in_weight_path = config['checkpoint']['in_qt_weight_path']
        self.out_checkpoint_path = config['checkpoint']['out_qt_checkpoint_path']
        self.out_weight_path = config['checkpoint']['out_qt_weight_path']

        embedding_path = config['dataset']['embedding_path']
        embedding_freeze = config['dataset']['embedding_freeze']
        self.model = DocRepQTTrainModel(config['model'], embedding_path, embedding_freeze)


class DAQTRep(BaseModule):
    def __init__(self, config):
        super(DAQTRep, self).__init__()
        self.name = 'qt'

        self.in_checkpoint_path = config['checkpoint']['in_qt_checkpoint_path']
        self.in_weight_path = config['checkpoint']['in_qt_weight_path']
        embedding_path = config['dataset']['embedding_path']
        embedding_freeze = config['dataset']['embedding_freeze']

        self.model = DocRepQTTestModel(config['model'], embedding_path, embedding_freeze)


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

    def __init__(self, model_config, embedding_path, embedding_freeze=True):
        super(DocRepQTTrainModel, self).__init__()
        embedding_weight = torch.tensor(np.load(embedding_path), dtype=torch.float32)
        logger.info('Embedding shape: ' + str(embedding_weight.shape))
        self.embedding_layer = torch.nn.Embedding.from_pretrained(embedding_weight,
                                                                  freeze=embedding_freeze,
                                                                  padding_idx=Vocabulary.padding_idx)

        self.tar_doc_encoder = DocRepQTEncoder(model_config)
        self.cand_doc_encoder = DocRepQTEncoder(model_config)

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
        document_rep: (batch, label_size, hidden_size * 4)
    """

    def __init__(self, model_config, embedding_path, embedding_freeze=True):
        super(DocRepQTTestModel, self).__init__()
        self.label_size = model_config['label_size']
        embedding_weight = torch.tensor(np.load(embedding_path), dtype=torch.float32)
        logger.info('Embedding shape: ' + str(embedding_weight.shape))
        self.embedding_layer = torch.nn.Embedding.from_pretrained(embedding_weight,
                                                                  freeze=embedding_freeze,
                                                                  padding_idx=-1)

        self.tar_doc_encoder = DocRepQTEncoder(model_config)
        self.cand_doc_encoder = DocRepQTEncoder(model_config)

    def forward(self, doc, doc_mask):
        # embedding layer
        doc = self.embedding_layer(doc)

        batch = doc.size(0)
        doc_rep = []

        for i in range(self.label_size):
            cur_label = doc.new_zeros(batch, self.label_size)
            cur_label[:, i] = 1

            # doc encoder layer
            tar_doc_rep, _ = self.tar_doc_encoder(doc, doc_mask, cur_label)
            cand_doc_rep, _ = self.cand_doc_encoder(doc, doc_mask, cur_label)

            # doc representation
            cur_doc_rep = torch.cat([tar_doc_rep, cand_doc_rep], dim=-1)
            doc_rep.append(cur_doc_rep)
        doc_rep = torch.stack(doc_rep, dim=1)

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

    def __init__(self, model_config):
        super(DocRepQTEncoder, self).__init__()

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
