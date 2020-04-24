#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""Multi-Label Classification"""

import json
import argparse
from tqdm import tqdm
import torch
import torch.nn
import torch.multiprocessing
import logging
from models import *
from datareaders import *
from utils.optims import Optim
from utils.metrics import *
from utils.config import init_logging, init_env

logger = logging.getLogger(__name__)


def main(config_path, in_infix, out_infix, is_train, is_test, gpuid):
    logger.info('-------------Multi-Label Classification---------------')
    logger.info('initial environment...')
    config, enable_cuda, device, writer = init_env(config_path, in_infix, out_infix,
                                                   writer_suffix='cls_log_path', gpuid=gpuid)

    logger.info('reading dataset...')
    # dataset = DocRepClsReader(config)
    dataset = DocClsReader(config)

    logger.info('constructing model...')
    # model = MultiLabelCls(config).to(device)
    model = E2EMultiLabelCls(config).to(device)
    model.load_parameters(enable_cuda)      # replace=('tar_doc_encoder', 'encoder')

    # loss function
    criterion = MultiLabelCls.criterion
    optimizer = Optim(config['train']['optimizer'],
                      lr=config['train']['learning_rate'],
                      max_grad_norm=config['train']['clip_grad_norm'],
                      lr_decay=config['train']['learning_rate_decay'],
                      start_decay_at=config['train']['start_decay_at'])
    optimizer.set_parameters(model.parameters())

    # dataset
    train_data = dataset.get_dataloader_train()
    valid_data = dataset.get_dataloader_valid()
    test_data = dataset.get_dataloader_test()

    if is_train:
        logger.info('start training...')

        num_epochs = config['train']['num_epochs']

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
                           device=device,
                           writer=writer)
            optimizer.updateLearningRate(epoch)  # learning rate decay

            # evaluate
            with torch.no_grad():
                model.eval()  # let training = False, make sure right dropout
                metrics, _ = eval_on_model(model=model,
                                           batch_data=valid_data,
                                           device=device)
            logger.info('epoch={}: '.format(epoch) + print_metrics(metrics, stage='valid'))

            # save best model with maximum micro-f1 and macro-f1
            if best_metrics is None or metrics['micro_f1'] + metrics['macro_f1'] > \
                    best_metrics['micro_f1'] + best_metrics['macro_f1']:
                model.save_parameters(epoch)

                best_metrics = metrics
                best_epoch = epoch
        logging.info('best epoch={}: '.format(best_epoch) + print_metrics(best_metrics, stage='valid'))
        with open('outputs/' + out_infix + '/valid_metrics.json', 'w') as wf:
            json.dump(best_metrics, wf, indent=2)

    if is_test:
        logger.info('start testing...')
        if is_train:
            model.load_out_parameters(enable_cuda, force=True, strict=True)

        with torch.no_grad():
            model.eval()
            metrics, predict = eval_on_model(model=model,
                                             batch_data=test_data,
                                             device=device)
        logger.info(print_metrics(metrics, stage='test'))
        with open('outputs/' + out_infix + '/metrics.json', 'w') as wf:
            json.dump(metrics, wf, indent=2)
        torch.save(predict, 'outputs/' + out_infix + '/predict.pt')

    writer.close()
    logger.info('finished.')


def print_metrics(metrics, stage='test'):
    out = "\n{stage}_macro_f1={macro_f1:.2%}," \
          "\n{stage}_micro_f1={micro_f1:.2%}," \
          "\n{stage}_micro_p={micro_p:.2%}," \
          "\n{stage}_micro_r={micro_r:.2%}," \
          "\n{stage}_hamming_loss={hl:.4f}," \
          "\n{stage}_one_error={oe:.2%}" \
        .format(stage=stage, macro_f1=metrics['macro_f1'], micro_f1=metrics['micro_f1'],
                micro_p=metrics['micro_p'], micro_r=metrics['micro_r'],
                hl=metrics['hamming_loss'], oe=metrics['one_error'])
    return out


def train_on_model(epoch, model, criterion, optimizer, batch_data, device, writer):
    batch_cnt = len(batch_data)
    sum_loss = 0.
    for i, batch in tqdm(enumerate(batch_data), total=batch_cnt, desc='Training Epoch=%d' % epoch):
        model.zero_grad()

        # batch data
        batch = [x.to(device) if x is not None else x for x in batch]
        truth = batch[-1]
        batch_input = batch[:-1]

        # forward
        predict = model.forward(*batch_input)

        loss = criterion(predict, truth)
        loss.backward()

        # evaluate
        macro_f1, micro_f1, micro_p, micro_r, _ = evaluate_f1_ml(predict, truth)
        hamming_loss = evaluate_hamming_loss(predict, truth)
        one_error = evaluate_one_error(predict, truth)

        optimizer.step()  # update parameters

        # logging
        batch_loss = loss.item()
        sum_loss += batch_loss * truth.shape[0]
        writer.add_scalar('Train-Step-Loss', batch_loss, global_step=epoch * batch_cnt + i)
        writer.add_scalar('Train-Step-Macro_F1', macro_f1, global_step=epoch * batch_cnt + i)
        writer.add_scalar('Train-Step-Micro_F1', micro_f1, global_step=epoch * batch_cnt + i)
        writer.add_scalar('Train-Step-Micro_P', micro_p, global_step=epoch * batch_cnt + i)
        writer.add_scalar('Train-Step-Micro_R', micro_r, global_step=epoch * batch_cnt + i)
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
    macro_f1, micro_f1, micro_p, micro_r, label_f1 = evaluate_f1_ml(predict, truth)
    hamming_loss = evaluate_hamming_loss(predict, truth)
    one_error = evaluate_one_error(predict, truth)

    metrics = {'macro_f1': macro_f1,
               'micro_f1': micro_f1,
               'micro_p': micro_p,
               'micro_r': micro_r,
               'hamming_loss': hamming_loss,
               'one_error': one_error,
               'label_f1': label_f1}
    return metrics, predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config.yaml', help='config path')
    parser.add_argument('-in', dest='in_infix', type=str, default='default', help='input data_path infix')
    parser.add_argument('-out', type=str, default='default', help='output data_path infix')
    parser.add_argument('-train', action='store_true', default=False, help='enable train step')
    parser.add_argument('-test', action='store_true', default=False, help='enable test step')
    parser.add_argument('-gpuid', type=int, default=None, help='gpuid')
    args = parser.parse_args()

    init_logging(out_infix=args.out)
    main(args.config, args.in_infix, args.out, is_train=args.train, is_test=args.test, gpuid=args.gpuid)
