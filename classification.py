#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""Multi-Label Classification"""

import argparse
from tqdm import tqdm
import torch
import torch.nn
import torch.multiprocessing
import logging
from models import MultiLabelCls
from datareaders import DocClsReader
from utils.functions import get_optimizer
from utils.metrics import *
from utils.config import init_logging, init_env

logger = logging.getLogger(__name__)


def main(config_path, in_infix, out_infix, is_train, is_test):
    logger.info('-------------Multi-Label Classification---------------')
    logger.info('initial environment...')
    config, enable_cuda, device, writer = init_env(config_path, in_infix, out_infix,
                                                   writer_suffix='cls_log_path')

    logger.info('reading dataset...')
    dataset = DocClsReader(config)

    logger.info('constructing model...')
    model = MultiLabelCls(config).to(device)
    model.load_parameters(enable_cuda)

    # loss function
    criterion = MultiLabelCls.criterion
    optimizer = get_optimizer(config['train']['optimizer'],
                              config['train']['learning_rate'],
                              model.parameters())

    # dataset
    train_data = dataset.get_dataloader_train()
    valid_data = dataset.get_dataloader_valid()
    test_data = dataset.get_dataloader_test()

    if is_train:
        logger.info('start training...')

        num_epochs = config['train']['num_epochs']
        clip_grad_max = config['train']['clip_grad_norm']

        best_metrics = None
        best_epoch = 0

        for epoch in range(1, num_epochs + 1):
            # train
            model.train()  # set training = True, make sure right dropout
            train_on_model(epoch=epoch,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           batch_data=train_data,
                           clip_grad_max=clip_grad_max,
                           device=device,
                           writer=writer)
            # evaluate
            with torch.no_grad():
                model.eval()  # let training = False, make sure right dropout
                metrics = eval_on_model(model=model,
                                        batch_data=valid_data,
                                        device=device)
            logger.info("epoch=%d, valid_macro_f1=%.2f%%, valid_micro_f1=%.2f%%, "
                        "valid_hamming_loss=%.3f, valid_one_error=%.2f%%" %
                        (epoch, metrics['macro_f1'] * 100, metrics['micro_f1'] * 100,
                         metrics['hamming_loss'], metrics['one_error'] * 100))

            if best_metrics is None or metrics['macro_f1'] > best_metrics['macro_f1']:
                model.save_parameters(epoch)

                best_metrics = metrics
                best_epoch = epoch
        logging.info('best epoch=%d: valid_macro_f1=%.2f%%, valid_micro_f1=%.2f%%, '
                     'valid_hamming_loss=%.3f, valid_one_error=%.2f%%'
                     % (best_epoch, best_metrics['macro_f1'] * 100, best_metrics['micro_f1'] * 100,
                        best_metrics['hamming_loss'], best_metrics['one_error'] * 100))

    if is_test:
        logger.info('start testing...')
        if is_train:
            model.load_out_parameters(enable_cuda, force=True, strict=True)

        with torch.no_grad():
            model.eval()
            metrics = eval_on_model(model=model,
                                    batch_data=test_data,
                                    device=device)
        logger.info("test_macro_f1=%.2f%%, test_micro_f1=%.2f%%, test_hamming_loss=%.3f, test_one_error=%.2f%%" %
                    (metrics['macro_f1'] * 100, metrics['micro_f1'] * 100,
                     metrics['hamming_loss'], metrics['one_error'] * 100))

    writer.close()
    logger.info('finished.')


def train_on_model(epoch, model, criterion, optimizer, batch_data, clip_grad_max,
                   device, writer):
    batch_cnt = len(batch_data)
    sum_loss = 0.
    for i, batch in tqdm(enumerate(batch_data), total=batch_cnt, desc='Training Epoch=%d' % epoch):
        optimizer.zero_grad()

        # batch data
        batch = [x.to(device) if x is not None else x for x in batch]
        truth = batch[-1]
        batch_input = batch[:-1]

        # forward
        predict = model.forward(*batch_input)

        loss = criterion(predict, truth)
        loss.backward()

        # evaluate
        macro_f1, micro_f1 = evaluate_f1_ml(predict, truth)
        hamming_loss = evaluate_hamming_loss(predict, truth)
        one_error = evaluate_one_error(predict, truth)

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_max)  # fix gradient explosion
        optimizer.step()  # update parameters

        # logging
        batch_loss = loss.item()
        sum_loss += batch_loss * truth.shape[0]
        writer.add_scalar('Train-Step-Loss', batch_loss, global_step=epoch * batch_cnt + i)
        writer.add_scalar('Train-Step-Macro_F1', macro_f1, global_step=epoch * batch_cnt + i)
        writer.add_scalar('Train-Step-Micro_F1', micro_f1, global_step=epoch * batch_cnt + i)
        writer.add_scalar('Train-Step-Hamming_Loss', hamming_loss, global_step=epoch * batch_cnt + i)
        writer.add_scalar('Train-Step-One_Error', one_error, global_step=epoch * batch_cnt + i)


def eval_on_model(model, batch_data, device):
    batch_cnt = len(batch_data)
    all_predict = []
    all_truth = []

    for i, batch in tqdm(enumerate(batch_data), total=batch_cnt, desc='Testing...'):
        # batch data
        batch = [x.to(device) if x is not None else x for x in batch]
        truth = batch[-1]
        all_truth.append(truth)

        batch_input = batch[:-1]

        # forward
        predict = model.forward(*batch_input)
        all_predict.append(predict)

    predict = torch.cat(all_predict, dim=0)
    truth = torch.cat(all_truth, dim=0)
    macro_f1, micro_f1 = evaluate_f1_ml(predict, truth)
    hamming_loss = evaluate_hamming_loss(predict, truth)
    one_error = evaluate_one_error(predict, truth)

    metrics = {'macro_f1': macro_f1,
               'micro_f1': micro_f1,
               'hamming_loss': hamming_loss,
               'one_error': one_error}
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='in_infix', type=str, default='default', help='input data_path infix')
    parser.add_argument('--out', type=str, default='default', help='output data_path infix')
    parser.add_argument('--train', action='store_true', default=False, help='enable train step')
    parser.add_argument('--test', action='store_true', default=False, help='enable test step')
    args = parser.parse_args()

    init_logging(out_infix=args.out)
    main('config/config.yaml', args.in_infix, args.out, is_train=args.train, is_test=args.test)
