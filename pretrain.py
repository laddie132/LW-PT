#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import argparse
import torch
import torch.nn
import logging
from tqdm import tqdm
from models import DAQT
from datareaders import RMSC
from utils.functions import get_optimizer
from utils.config import init_logging, init_env

logger = logging.getLogger(__name__)


def main(config_path, in_infix, out_infix, is_train, is_test):
    logger.info('-------------DA-QT Pre-Training---------------')
    logger.info('initial environment...')
    game_config, enable_cuda, device, writer = init_env(config_path, in_infix, out_infix,
                                                        writer_suffix='qt_log_path')
    logger.info('reading dataset...')
    dataset = RMSC(game_config)

    logger.info('constructing model...')
    model = DAQT(game_config).to(device)
    model.load_parameters(enable_cuda)

    # loss function
    criterion = torch.nn.NLLLoss()
    optimizer = get_optimizer(game_config['train']['optimizer'],
                              game_config['train']['learning_rate'],
                              model.parameters())

    # training arguments
    batch_size = game_config['train']['batch_size']
    num_workers = game_config['global']['num_data_workers']

    # dataset loader
    batch_train_data = dataset.get_dataloader_train(batch_size, num_workers)
    batch_test_data = dataset.get_dataloader_test(batch_size, num_workers)

    if is_train:
        logger.info('start training...')

        clip_grad_max = game_config['train']['clip_grad_norm']
        num_epochs = game_config['train']['num_epochs']
        save_steps = game_config['train']['save_steps']

        # train
        model.train()  # set training = True, make sure right dropout
        for epoch in range(num_epochs):
            train_on_model(epoch=epoch,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           batch_data=batch_train_data,
                           clip_grad_max=clip_grad_max,
                           device=device,
                           writer=writer,
                           save_steps=save_steps)

    if is_test:
        logger.info('start testing...')

        with torch.no_grad():
            model.eval()
            metrics = eval_on_model(model=model,
                                    batch_data=batch_test_data,
                                    device=device)
        logger.info("Acc=%.3f, P_1=%.3f, P_3=%.3f, MAP=%.3f" %
                    (metrics['docs_acc'], metrics['top_p1'], metrics['top_p2'], metrics['map']))

    writer.close()
    logger.info('finished.')


def train_on_model(epoch, model, criterion, optimizer, batch_data, clip_grad_max, device, writer, save_steps):
    pass


def eval_on_model(model, batch_data, device):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='in_infix', type=str, default='default', help='input data_path infix')
    parser.add_argument('--out', type=str, default='default', help='output data_path infix')
    parser.add_argument('--train', action='store_true', default=False, help='enable train step')
    parser.add_argument('--test', action='store_true', default=False, help='enable test step')
    args = parser.parse_args()

    init_logging(out_infix=args.out)
    main('config/game_config.yaml', args.in_infix, args.out, is_train=args.train, is_test=args.test)