#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""pre-build documents representation"""

import argparse
from tqdm import tqdm
import torch
import torch.nn
import torch.multiprocessing
import logging
from models import DAQTRep
from datareaders import QTRepReader
from utils.config import init_logging, init_env

logger = logging.getLogger(__name__)


def main(config_path, in_infix, out_infix):
    logger.info('-------------Doc-Rep Pre-building---------------')
    logger.info('initial environment...')
    config, enable_cuda, device, writer = init_env(config_path, in_infix, out_infix,
                                                   writer_suffix='qt_log_path')

    logger.info('reading dataset...')
    dataset = QTRepReader(config)

    logger.info('constructing model...')
    doc_rep_module = DAQTRep(config).to(device)
    doc_rep_module.load_parameters(enable_cuda, force=True, strict=False)

    # dataset
    train_data = dataset.get_dataloader_train()
    valid_data = dataset.get_dataloader_valid()
    test_data = dataset.get_dataloader_test()

    with torch.no_grad():
        logger.info('start documents encoding...')
        doc_rep_module.eval()
        train_doc_rep = test_on_model(doc_rep_module, train_data, device)
        valid_doc_rep = test_on_model(doc_rep_module, valid_data, device)
        test_doc_rep = test_on_model(doc_rep_module, test_data, device)

        logger.info('saving documents vectors...')
        torch.save({'train_doc_rep': train_doc_rep,
                    'valid_doc_rep': valid_doc_rep,
                    'test_doc_rep': test_doc_rep}, config['dataset']['doc_rep_path'])

    logger.info('finished.')


def test_on_model(model, dataloader, device):
    all_doc_rep = []

    for batch in tqdm(dataloader, desc='Building...'):
        batch = [x.to(device) if x is not None else x for x in batch]

        # forward
        batch_doc_rep = model.forward(*batch)
        all_doc_rep.append(batch_doc_rep)

    all_doc_rep = torch.cat(all_doc_rep, dim=0)
    return all_doc_rep


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='in_infix', type=str, default='default', help='input path infix')
    parser.add_argument('--out', type=str, default='default', help='output path infix')
    args = parser.parse_args()

    init_logging(out_infix=args.out)
    main('config/config.yaml', in_infix=args.in_infix, out_infix=args.out)
