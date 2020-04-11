#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import torch
import torch.nn
import logging
from .base import BaseModule
from . import encoder
from . import decoder
from datareaders.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


class MultiLabelCls(BaseModule):
    def __init__(self, config):
        super(MultiLabelCls, self).__init__()
        self.in_checkpoint_path = config['checkpoint']['in_cls_checkpoint_path']
        self.in_weight_path = config['checkpoint']['in_cls_weight_path']
        self.out_checkpoint_path = config['checkpoint']['out_cls_checkpoint_path']
        self.out_weight_path = config['checkpoint']['out_cls_weight_path']

        self.model = getattr(decoder, config['model']['decoder'])(config['model'])

    @staticmethod
    def criterion(y_pred, y_true, reduction='mean'):
        """
        a personal negative log likelihood loss. It is useful to train a classification problem with `C` classes.
        :param y_pred: (batch, labels)
        :param y_true: (batch, labels), 0 or 1
        :param reduction:
        :return:
        """
        y_pred_log = torch.log(y_pred)
        non_y_pred_log = torch.log(1 - y_pred)
        valid_docs_prob_log = y_pred_log * y_true.float()
        non_docs_prob_log = non_y_pred_log * (1 - y_true).float()
        batch_loss = -valid_docs_prob_log.sum(dim=-1) - non_docs_prob_log.sum(dim=-1)

        if reduction == 'none':
            return batch_loss
        elif reduction == 'sum':
            return batch_loss.sum()
        elif reduction == 'mean':
            return batch_loss.sum() / batch_loss.shape[0]
        else:
            raise ValueError(reduction)


class E2EMultiLabelCls(BaseModule):
    def __init__(self, config):
        super(E2EMultiLabelCls, self).__init__()
        self.in_checkpoint_path = config['checkpoint']['in_cls_checkpoint_path']
        self.in_weight_path = config['checkpoint']['in_cls_weight_path']
        self.out_checkpoint_path = config['checkpoint']['out_cls_checkpoint_path']
        self.out_weight_path = config['checkpoint']['out_cls_weight_path']

        embedding_path = config['dataset']['embedding_path']
        embedding_freeze = config['dataset']['embedding_freeze']

        self.model = E2EMLCModel(config['model'], embedding_path, embedding_freeze)


class E2EMLCModel(torch.nn.Module):
    def __init__(self, model_config, embedding_path, embedding_freeze):
        super(E2EMLCModel, self).__init__()
        embedding_num = model_config['embedding_num']
        embedding_dim = model_config['embedding_dim']

        if not model_config['use_pretrain']:
            self.embedding_layer = torch.nn.Embedding(num_embeddings=embedding_num,
                                                      embedding_dim=embedding_dim,
                                                      padding_idx=Vocabulary.PAD_IDX)
        else:
            embedding_weight = Vocabulary.load_emb(embedding_path)
            logger.info('Embedding shape: ' + str(embedding_weight.shape))
            self.embedding_layer = torch.nn.Embedding.from_pretrained(embedding_weight,
                                                                      freeze=embedding_freeze,
                                                                      padding_idx=Vocabulary.PAD_IDX)

        self.encoder = getattr(encoder, model_config['encoder'])(model_config)
        self.decoder = getattr(decoder, model_config['decoder'])(model_config)

    def forward(self, doc, *args):
        doc_emb = self.embedding_layer(doc)
        return self.decoder(self.encoder(doc_emb, *args)[0])
