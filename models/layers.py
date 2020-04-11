#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""Personalizerd PyTorch Layers"""

import math
import torch
import torch.nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from utils.functions import masked_softmax, compute_top_layer_mask


class HierarchicalAttention(torch.nn.Module):
    """
    Hierarchical LinearAttention Layer
    Args:
        hidden_size: hidden size
    Inputs:
        x (batch, top_len, sub_len, hidden_size):
        x_mask (batch, top_len, sub_len): mask of input
    Outputs:
        x_top_att_rep (batch, hidden_size):
        x_top_prop (batch, top_len):
    """
    def __init__(self, hidden_size, dropout_p):
        super(HierarchicalAttention, self).__init__()
        self.sub_attention = LinearAttention(hidden_size, dropout_p)
        self.top_attention = LinearAttention(hidden_size, dropout_p)

    def forward(self, x, x_mask, q=None):
        # sub layer
        batch, top_len, sub_len, hidden_size = x.size()
        x_flip = x.view(-1, sub_len, hidden_size)
        x_flip_mask = x_mask.view(-1, sub_len)
        x_flip_att_rep, _ = self.sub_attention(x_flip, x_flip_mask, q)

        x_sub_att_rep = x_flip_att_rep.view(batch, top_len, hidden_size)  # (batch, top_len, hidden_size)

        # top layer
        x_top_mask = compute_top_layer_mask(x_mask)
        x_top_att_rep, x_top_prop = self.top_attention(x_sub_att_rep, x_top_mask, q)

        return x_top_att_rep, x_top_prop


class LinearAttention(torch.nn.Module):
    """
    Linear + Attention layer
    Args:
        hidden_size: hidden size
    Inputs:
        x: (batch, len, hidden_size)
        x_mask: (batch, len)
        q: (batch, hidden_size), self-attention if q is None
    Outputs:
        x_att_rep: (batch, hidden_size)
        x_prop: (batch, len)
    """
    def __init__(self, hidden_size, dropout_p):
        super(LinearAttention, self).__init__()

        self.self_linear = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout_layer = torch.nn.Dropout(p=dropout_p)
        self.alpha_linear = torch.nn.Linear(hidden_size, 1)
        self.query_linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x, x_mask=None, q=None):
        if q is not None:
            # (batch, 1, hidden_size)
            q_rep = self.query_linear(q).unsqueeze(1)
            x_tanh = torch.tanh(self.self_linear(x) + q_rep)
        else:
            x_tanh = torch.tanh(self.self_linear(x))

        x_tanh = self.dropout_layer(x_tanh)

        x_alpha = self.alpha_linear(x_tanh) \
            .squeeze(-1)  # (batch, len)

        x_prop = masked_softmax(x_alpha, x_mask, dim=-1)
        x_att_rep = torch.bmm(x_prop.unsqueeze(1), x) \
            .squeeze(1)  # (batch, hidden_size)

        return x_att_rep, x_prop


class SelfAttention(torch.nn.Module):
    """
    Self Attention layer
    Args:
        hidden_size: hidden size
    Inputs:
        x: (batch, len, hidden_size)
        x_mask: (batch, len)
    Outputs:
        x_att_rep: (batch, hidden_size)
        x_prop: (batch, len)
    """

    def __init__(self, in_features):
        super(SelfAttention, self).__init__()

        self.alpha_linear = torch.nn.Linear(in_features, 1)

    def forward(self, x, x_mask=None):
        x_alpha = self.alpha_linear(x) \
            .squeeze(-1)  # (batch, len)

        x_prop = masked_softmax(x_alpha, x_mask, dim=-1)
        x_att_rep = torch.bmm(x_prop.unsqueeze(1), x) \
            .squeeze(1)  # (batch, hidden_size)

        return x_att_rep, x_prop


class MultiHeadSelfAttention(torch.nn.Module):
    """
    Self Attention layer with multi-head
    Args:
        in_features: hidden size
    Inputs:
        x: (batch, len, hidden_size)
        label: (batch, label_size)
        x_mask: (batch, len)
    Outputs:
        x_att_rep: (batch, hidden_size)
        x_prop: (batch, len)
    """

    def __init__(self, in_features, labels, bias=True):
        super(MultiHeadSelfAttention, self).__init__()
        self.weight = Parameter(torch.Tensor(labels, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(labels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, label, x_mask=None):
        cur_weight = torch.mm(label.float(), self.weight).unsqueeze(-1)     # (batch, in_features, 1)
        cur_bias = None
        if self.bias is not None:
            cur_bias = torch.mm(label.float(), self.bias.unsqueeze(-1))     # (batch, 1)

        x_alpha = torch.bmm(x, cur_weight).squeeze(-1)      # (batch, len)
        if cur_bias is not None:
            x_alpha += cur_bias

        x_prop = masked_softmax(x_alpha, x_mask, dim=-1)
        x_att_rep = torch.bmm(x_prop.unsqueeze(1), x) \
            .squeeze(1)  # (batch, hidden_size)

        return x_att_rep, x_prop


class MyRNNBase(torch.nn.Module):
    """
    RNN with packed sequence and dropout, only one layer
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        dropout_p: dropout probability to input data, and also dropout along hidden layers
        enable_layer_norm: layer normalization
        batch_first: (batch, seq_len, ...) or (seq_len, batch, ...)
    Inputs:
        input (batch, seq_len, input_size): tensor containing the features of the input sequence.
        mask (batch, seq_len): tensor show whether a padding index for each element in the batch.
    Outputs:
        output (seq_len, batch, hidden_size * num_directions): tensor containing the output
            features `(h_t)` from the last layer of the RNN, for each t.
        last_state (batch, hidden_size * num_directions): the final hidden state of rnn
    """

    def __init__(self, mode, input_size, hidden_size, bidirectional, dropout_p,
                 enable_layer_norm=False, batch_first=True, num_layers=1):
        super(MyRNNBase, self).__init__()
        self.mode = mode
        self.num_layers = num_layers
        self.enable_layer_norm = enable_layer_norm
        self.batch_first = batch_first

        if mode == 'LSTM':
            self.hidden = torch.nn.LSTM(input_size=input_size,
                                        hidden_size=hidden_size,
                                        dropout=dropout_p,
                                        num_layers=num_layers,
                                        bidirectional=bidirectional)
        elif mode == 'GRU':
            self.hidden = torch.nn.GRU(input_size=input_size,
                                       hidden_size=hidden_size,
                                       dropout=dropout_p,
                                       num_layers=num_layers,
                                       bidirectional=bidirectional)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.dropout = torch.nn.Dropout(p=dropout_p)

        if enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(input_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, v, mask):
        if self.batch_first:
            v = v.transpose(0, 1)

        # layer normalization
        if self.enable_layer_norm:
            seq_len, batch, input_size = v.shape
            v = v.contiguous().view(-1, input_size)
            v = self.layer_norm(v)
            v = v.view(seq_len, batch, input_size)

        # get sorted v
        lengths = mask.eq(1).long().sum(1)
        lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        v_sort = v.index_select(1, idx_sort)

        # remove zeros lengths
        zero_idx = lengths_sort.nonzero()[-1][0].item() + 1
        zeros_len = lengths_sort.shape[0] - zero_idx

        lengths_sort = lengths_sort[:zero_idx]
        v_sort = v_sort[:, :zero_idx, :]

        # rnn
        v_pack = torch.nn.utils.rnn.pack_padded_sequence(v_sort, lengths_sort)
        v_dropout = self.dropout.forward(v_pack.data)
        v_pack_dropout = torch.nn.utils.rnn.PackedSequence(v_dropout, v_pack.batch_sizes)

        o_pack_dropout, o_last = self.hidden.forward(v_pack_dropout)
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o_pack_dropout)

        # get the last time state
        if isinstance(o_last, tuple):
            o_last = o_last[0]    # if LSTM cell used

        _, batch, hidden_size = o_last.size()
        o_last = o_last.view(self.num_layers, -1, batch, hidden_size)
        o_last = o_last[-1, :].transpose(0, 1).contiguous().view(batch, -1)

        # len_idx = (lengths_sort - 1).view(-1, 1).expand(-1, o.size(2)).unsqueeze(0)
        # o_last = o.gather(0, len_idx)
        # o_last = o_last.squeeze(0)

        # padding for output and output last state
        if zeros_len > 0:
            o_padding_zeros = o.new_zeros(o.shape[0], zeros_len, o.shape[2])
            o = torch.cat([o, o_padding_zeros], dim=1)

            o_last_padding_zeros = o_last.new_zeros(zeros_len, o_last.shape[1])
            o_last = torch.cat([o_last, o_last_padding_zeros], dim=0)

        # unsorted o
        o_unsort = o.index_select(1, idx_unsort)  # Note that here first dim is seq_len
        o_last_unsort = o_last.index_select(0, idx_unsort)

        if self.batch_first:
            o_unsort = o_unsort.transpose(0, 1)

        return o_unsort, o_last_unsort


class TransformerModel(torch.nn.Module):
    """
    Transformer model with position encoding
    Args:
        nemb: embeddings size
        nhead: number of attention heads
        nhid: hidden size
        nlayers: number of transformer layers
        dropout: dropout probability
    Inputs:
        src (batch, seq_len, nemb): tensor containing the features of the input sequence.
        src_key_value_mask (batch, seq_len): tensor show whether a padding index for each element in the batch.
    Outputs:
        output (batch, nhid)
    """
    def __init__(self, nemb, nhead, nhid, nlayers, dropout):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.nemb = nemb

        self.pos_encoder = PositionalEncoding(nemb, dropout)
        encoder_layers = torch.nn.TransformerEncoderLayer(nemb, nhead, nhid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)

        self.doc_attention = SelfAttention(in_features=nemb)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).float()    # must be float
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_key_value_mask):
        visual_parm = {}

        src = src.transpose(0, 1)
        src_key_padding_mask = (1 - src_key_value_mask).bool()    # mask different with input

        # only allowed to attend the earlier positions in the sequence
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     device = src.device
        #     mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #     self.src_mask = mask

        src = src * math.sqrt(self.nemb)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=self.src_mask, src_key_padding_mask=src_key_padding_mask)

        output = output.transpose(0, 1)
        doc_rep, doc_att_p = self.doc_attention(output, src_key_value_mask)

        visual_parm['doc_att_p'] = doc_att_p

        return doc_rep, visual_parm


class PositionalEncoding(torch.nn.Module):
    """
    Position Encoding
    Args:
        d_model: embeddings size
        dropout: dropout probability
        max_len: sentence max length
    Inputs:
        x (seq_len, batch, nemb): tensor containing the features of the input sequence.
    Outputs:
        output (seq_len, batch, nemb)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return NotImplementedError
        x = x + self.pe[x.size(0), :]
        return self.dropout(x)
