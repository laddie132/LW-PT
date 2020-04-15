#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""
Pre-Processing the dataset
PYTHONHASHSEED should be set to 1 before running
"""

import argparse
import preprocessors
from utils.functions import set_seed
from utils.config import init_logging

random_seed = 1


def run(data_name, data_path):
    dataset = getattr(preprocessors, data_name)(data_path=data_path, random_seed=random_seed)
    dataset.build()
    set_seed(random_seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default='', help='dataset name')
    parser.add_argument('-path', type=str, default='', help='dataset path')
    args = parser.parse_args()

    init_logging()
    run(args.data, args.path)
