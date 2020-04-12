#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""DA-QT Module"""

import torch
import logging
from .base import BaseModule
from . import encoder
from .layers import get_embedding_layer

logger = logging.getLogger(__name__)


class DAQT(BaseModule):
    def __init__(self, config):
        super(DAQT, self).__init__(
            config,
            name='qt',
            model=DocRepQTTrainModel(config['model']))


class DAQTRep(BaseModule):
    def __init__(self, config):
        super(DAQTRep, self).__init__(
            config,
            name='qt',
            model=DocRepQTTestModel(config['model']))


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
        self.embedding_layer = get_embedding_layer(model_config)
        self.tar_doc_encoder = getattr(encoder, model_config['encoder'])(model_config)
        self.cand_doc_encoder = getattr(encoder, model_config['encoder'])(model_config)

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

    def __init__(self, model_config):
        super(DocRepQTTestModel, self).__init__()
        self.label_size = model_config['label_size']
        self.embedding_layer = get_embedding_layer(model_config)
        self.tar_doc_encoder = getattr(encoder, model_config['encoder'])(model_config)
        self.cand_doc_encoder = getattr(encoder, model_config['encoder'])(model_config)

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
