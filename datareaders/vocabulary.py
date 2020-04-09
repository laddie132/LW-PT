#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import torch
import numpy as np


class Vocabulary:
    padding_idx = 0

    @staticmethod
    def load_emb(embedding_path):
        emb = torch.tensor(np.load(embedding_path), dtype=torch.float32)
        add_emb = torch.zeros((1, emb.size(1)), dtype=torch.float)
        emb = torch.cat([add_emb, emb], dim=0)

        return emb
