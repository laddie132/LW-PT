#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

from .layers import *


class LWBiRNNEncoder(torch.nn.Module):
    """
    Label-Wise Bidirectional RNN encoder on word-level for document representation
    Inputs:
        doc_emb: (batch, doc_len, emb_dim)
        doc_mask: (batch, doc_len)
        label: (batch, label_size) or None
    Outputs:
        doc_rep: (batch, hidden_size * 2) / (batch, label_size, hidden_size * 2)
    """
    def __init__(self, model_config):
        super(LWBiRNNEncoder, self).__init__()

        embedding_dim = model_config['embedding_dim']
        hidden_size = model_config['hidden_size']
        label_size = model_config['label_size']
        dropout_p = model_config['dropout_p']
        enable_layer_norm = model_config['layer_norm']
        cell = model_config['cell']
        num_layers = model_config['num_layers']

        self.doc_rnn = MyRNNBase(mode=cell,
                                 input_size=embedding_dim,
                                 hidden_size=hidden_size,
                                 bidirectional=True,
                                 dropout_p=dropout_p,
                                 enable_layer_norm=enable_layer_norm,
                                 batch_first=True,
                                 num_layers=num_layers)
        self.doc_attention = MultiHeadSelfAttention(in_features=hidden_size * 2,
                                                    labels=label_size)

    def forward(self, doc_emb, doc_mask, label=None):
        visual_parm = {}

        # (batch, doc_len, hidden_size * 2)
        doc_rep, _ = self.doc_rnn(doc_emb, doc_mask)

        # (batch, hidden_size * 2) / (batch, label_size, hidden_size * 2)
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
        cell = model_config['cell']
        num_layers = model_config['num_layers']

        self.doc_word_rnn = MyRNNBase(mode=cell,
                                      input_size=embedding_dim,
                                      hidden_size=hidden_size,
                                      bidirectional=True,
                                      dropout_p=dropout_p,
                                      enable_layer_norm=enable_layer_norm,
                                      batch_first=True,
                                      num_layers=num_layers)
        self.doc_word_attention = MultiHeadSelfAttention(in_features=hidden_size * 2,
                                                         labels=label_size)

        self.doc_sentence_rnn = MyRNNBase(mode=cell,
                                          input_size=hidden_size * 2,
                                          hidden_size=hidden_size,
                                          bidirectional=True,
                                          dropout_p=dropout_p,
                                          enable_layer_norm=enable_layer_norm,
                                          batch_first=True,
                                          num_layers=num_layers)
        self.doc_sentence_attention = MultiHeadSelfAttention(in_features=hidden_size * 2,
                                                             labels=label_size)

    def forward(self, doc_emb, doc_mask, label=None):
        visual_parm = {}
        batch, doc_sent_len, doc_word_len, _ = doc_emb.size()

        doc_word_emb = doc_emb.view(batch * doc_sent_len, doc_word_len, -1)
        doc_word_mask = doc_mask.view(batch * doc_sent_len, doc_word_len)

        if label is not None:
            word_level_label = label.unsqueeze(1).expand(batch, doc_sent_len, -1).contiguous()
            word_level_label = word_level_label.view(batch * doc_sent_len, -1)
        else:
            word_level_label = None

        # (batch * doc_sent_len, doc_word_len, hidden_size * 2)
        doc_word_rep, _ = self.doc_word_rnn(doc_word_emb, doc_word_mask)

        # (batch * doc_sent_len, hidden_size * 2) / (batch * doc_sent_len, label_size, hidden_size * 2)
        doc_sent_emb, doc_word_att_p = self.doc_word_attention(doc_word_rep, word_level_label, doc_word_mask)
        visual_parm['doc_word_att_p'] = doc_word_att_p

        doc_sent_mask = compute_top_layer_mask(doc_mask)    # TODO: mismatch for aapd hier
        if label is not None:
            # (batch, doc_sent_len, hidden_size * 2)
            doc_sent_emb = doc_sent_emb.view(batch, doc_sent_len, -1)
        else:
            # (batch * label_size, doc_sent_len, hidden * 2)
            label_size = doc_sent_emb.shape[1]
            doc_sent_emb = doc_sent_emb.view(batch, doc_sent_len, label_size, -1).transpose(1, 2)
            doc_sent_emb = doc_sent_emb.reshape(batch * label_size, doc_sent_len, -1)

            doc_sent_mask = doc_sent_mask.unsqueeze(1).expand(-1, label_size, -1)
            doc_sent_mask = doc_sent_mask.reshape(batch * label_size, -1)

        # (batch, doc_sent_len, hidden_size * 2) / (batch * label_size, doc_sent_len, hidden_size * 2)
        doc_sent_rep, _ = self.doc_sentence_rnn(doc_sent_emb, doc_sent_mask)
        doc_sent_len = doc_sent_rep.shape[1]
        doc_sent_mask = doc_sent_mask[:, :doc_sent_len]

        # (batch, hidden_size * 2) / (batch * label_size, label_size, hidden_size * 2)
        doc_rep, doc_sent_att_p = self.doc_sentence_attention(doc_sent_rep, label, doc_sent_mask)
        if label is None:
            label_size = doc_rep.shape[1]
            hsize = doc_rep.shape[-1]
            doc_rep = doc_rep.view(batch, label_size, label_size, -1)
            select_idx = torch.eye(label_size, device=doc_rep.device).unsqueeze(-1).\
                repeat(batch, 1, 1, hsize).bool()
            doc_rep = doc_rep[select_idx].view(batch, label_size, -1)    # (batch, label_size, h * 2)

            doc_sent_att_p = doc_sent_att_p.view(batch, label_size, label_size, -1)
            select_idx = torch.eye(label_size, device=doc_rep.device).unsqueeze(-1).\
                repeat(batch, 1, 1, doc_sent_len).bool()
            doc_sent_att_p = doc_sent_att_p[select_idx].view(batch, label_size, -1)  # (batch, label_size, len)

        visual_parm['doc_sent_att_p'] = doc_sent_att_p

        return doc_rep, visual_parm


class BiRNNEncoder(torch.nn.Module):
    """
    Bidirectional RNN encoder on word-level for document representation
    Inputs:
        doc_emb: (batch, doc_len, emb_dim)
        doc_mask: (batch, doc_len)
        label: (batch, label_size)
    Outputs:
        doc_rep: (batch, hidden_size * 2)
    """
    def __init__(self, model_config):
        super(BiRNNEncoder, self).__init__()

        embedding_dim = model_config['embedding_dim']
        hidden_size = model_config['hidden_size']
        dropout_p = model_config['dropout_p']
        enable_layer_norm = model_config['layer_norm']
        cell = model_config['cell']
        num_layers = model_config['num_layers']

        self.doc_rnn = MyRNNBase(mode=cell,
                                 input_size=embedding_dim,
                                 hidden_size=hidden_size,
                                 bidirectional=True,
                                 dropout_p=dropout_p,
                                 enable_layer_norm=enable_layer_norm,
                                 batch_first=True,
                                 num_layers=num_layers)
        self.doc_attention = SelfAttention(in_features=hidden_size * 2)

    def forward(self, doc_emb, doc_mask, label=None):
        visual_parm = {}

        # (batch, doc_len, hidden_size * 2)
        doc_rep, _ = self.doc_rnn(doc_emb, doc_mask)

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
        cell = model_config['cell']
        num_layers = model_config['num_layers']

        self.doc_word_rnn = MyRNNBase(mode=cell,
                                      input_size=embedding_dim,
                                      hidden_size=hidden_size,
                                      bidirectional=True,
                                      dropout_p=dropout_p,
                                      enable_layer_norm=enable_layer_norm,
                                      batch_first=True,
                                      num_layers=num_layers)
        self.doc_word_attention = SelfAttention(in_features=hidden_size * 2)

        self.doc_sentence_rnn = MyRNNBase(mode=cell,
                                          input_size=hidden_size * 2,
                                          hidden_size=hidden_size,
                                          bidirectional=True,
                                          dropout_p=dropout_p,
                                          enable_layer_norm=enable_layer_norm,
                                          batch_first=True,
                                          num_layers=num_layers)
        self.doc_sentence_attention = SelfAttention(in_features=hidden_size * 2)

    def forward(self, doc_emb, doc_mask, label=None):
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
        doc_sent_len = doc_sent_rep.shape[1]
        doc_sent_mask = doc_sent_mask[:, :doc_sent_len]

        # (batch, hidden_size * 2)
        doc_rep, doc_sent_att_p = self.doc_sentence_attention(doc_sent_rep, doc_sent_mask)
        visual_parm['doc_sent_att_p'] = doc_sent_att_p

        return doc_rep, visual_parm


class CNNEncoder(torch.nn.Module):

    def __init__(self, model_config):
        super(CNNEncoder, self).__init__()
        embedding_dim = model_config['embedding_dim']
        dropout_p = model_config['dropout_p']
        Co = 100
        Ks = [3, 4, 5]

        self.convs1 = torch.nn.ModuleList([torch.nn.Conv2d(1, Co, (K, embedding_dim)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = torch.nn.Dropout(dropout_p)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, *args):
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        return x, None
