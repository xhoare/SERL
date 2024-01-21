#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse

import torch

from data_loader.dataloader import AudioDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.modules import Base_model
from trainer import rtrainer

import logging
from logger import set_logger


parser = argparse.ArgumentParser()
# General config
# Task related
parser.add_argument("--train_noisy_data_path", default='/data/noisy_trainset_mix_wav_16k')
parser.add_argument("--train_clean_data_path", default='/data/clean_trainset_mix_wav_16k_1s')
parser.add_argument("--train_file", default='/data/train.lst')
parser.add_argument("--valid_noisy_data_path", default='/data/noisy_validset_mix_wav_16k_1s')
parser.add_argument("--valid_clean_data_path", default='/data/clean_validset_mix_wav_16k_1s')
parser.add_argument("--valid_file", default='/data/valid.lst')
parser.add_argument("--sample_rate", type=int, default=16000)
parser.add_argument("--scale", type=int, default=0)
parser.add_argument("--low_pass", type=int, default=0)
parser.add_argument("--interpolate", type=int, default=0)
# Network architecture
parser.add_argument('--model', default=1, type=int)
parser.add_argument('--in_nc', default=1, type=int)
parser.add_argument('--out_nc', default=1, type=int)
parser.add_argument('--nf', default=128, type=int)
parser.add_argument('--ns', default=64, type=int)
parser.add_argument('--times', default=4, type=int)
parser.add_argument('--sinc', default=0, type=int)
parser.add_argument('--weights', default=0, type=float)
parser.add_argument('--mul', default=0, type=int)
parser.add_argument('--normalize', default=0, type=int)
parser.add_argument('--episod', default=0, type=int)
# Training config
parser.add_argument('--train_gpuid', default=[0],
                    help='Whether use GPU')
parser.add_argument('--epochs', default=300, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--early_stop', dest='early_stop', default=10, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=3, type=float,
                    help='Gradient norm threshold to clip')
parser.add_argument('--resume_state', default=0, type=int)
parser.add_argument('--resume_path', default='')
# minibatch
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size')
parser.add_argument('--num_workers', default=6, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')
parser.add_argument('--scheduler_factor', default=0.5, type=float)
parser.add_argument('--scheduler_patience', default=2, type=int)
parser.add_argument('--scheduler_min_lr', default=1e-8, type=float)
# save and load model
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
# logging
parser.add_argument('--print_freq', default=1000, type=int,
                    help='Frequency of printing training infomation')
parser.add_argument('--logger_name', default='asuper')
parser.add_argument('--logger_path', default='exp/log')
parser.add_argument('--logger_screen', default=False)
parser.add_argument('--logger_tofile', default=True)

def main(args):
    set_logger.setup_logger(args.logger_name, args.logger_path,
                            screen=args.logger_screen, tofile=args.logger_tofile)
    logger = logging.getLogger(args.logger_name)
    
    logger.info(args) 
    # build dataloader	
    logger.info('Building the dataloader')    
    train_dataset = AudioDataset(
        noisy_data_path = args.train_noisy_data_path,
        clean_data_path = args.train_clean_data_path,
        file_names = args.train_file,
        sampling_rate = args.sample_rate)

    val_dataset = AudioDataset(
        noisy_data_path = args.valid_noisy_data_path,
        clean_data_path = args.valid_clean_data_path,
        file_names = args.valid_file,
        sampling_rate=args.sample_rate)

    logger.info(val_dataset[0][0].size())
    logger.info(val_dataset[0][1].size())
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
    
    logger.info('Train Datasets Length: {}, Val Datasets Length: {}'.format(
        len(train_dataloader), len(val_dataloader)))
    
    # build model
    logger.info("Building the model")
    super_model = Base_model(in_nc=args.in_nc, out_nc=args.out_nc, nf=args.nf, gc=args.ns, times=args.times, normalize=args.normalize)
    logger.info(super_model)
    
    # build optimizer
    logger.info("Building the optimizer")
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(super_model.parameters(), 
                                     lr=args.lr, 
                                     weight_decay=args.l2
                                     )
    else:
        assert 1 == 0
    
    # build scheduler
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=args.scheduler_factor,
                                  patience=args.scheduler_patience,
                                  verbose=True, 
                                  min_lr=args.scheduler_min_lr)


    # build trainer
    logger.info('Builing the Trainer')
    logger.info('rtrainer')
    trainer = rtrainer.Trainer(train_dataloader, val_dataloader, super_model, optimizer, scheduler, opt=args)
    trainer.run()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)

