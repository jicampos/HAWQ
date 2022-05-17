import os
import json
import logging
from datetime import datetime 

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

from quant_train import train
from quant_train import validate
from quant_train import save_checkpoint
from quant_train import adjust_learning_rate

import utils 
from args import *
from utils import *
from bit_config import *


quantize_arch_dict = {'jettagger': utils.models.q_jettagger.jettagger_model,
                      'hawq_jettagger': utils.models.q_jettagger.q_jettagger_model,
                      'brevitas_jettagger': None} #utils.models.q_jettagger.BrevitasJetTagger}


train_loader = utils.getTrainData(dataset='hlc_jets',
                                    batch_size=args.batch_size,
                                    path='/data1/jcampos/HAWQ-main/data/train',
                                    for_inception=False,
                                    data_percentage=1)
val_loader = utils.getTestData(dataset='hlc_jets',
                                    batch_size=args.batch_size,
                                    path='/data1/jcampos/HAWQ-main/data/val',
                                    data_percentage=1)


def main(bit_config_key=None):

    now = datetime.now() # current date and time
    date_time = now.strftime('%m%d%Y_%H%M%S')

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if not os.path.exists(os.path.join(args.save_path, args.quant_scheme, date_time)):
        os.makedirs(os.path.join(args.save_path, args.quant_scheme, date_time))

    save_path = os.path.join(args.save_path, args.quant_scheme, date_time) + '/'
    logging.basicConfig(format='%(asctime)s - %(message)s', filemode='w', 
                    datefmt='%d-%b-%y %H:%M:%S', filename=save_path+'training.log', 
                    encoding='utf-8', level=logging.DEBUG)
    logging.info(args)

    model = quantize_arch_dict[args.arch](model=None)

    if args.resume:
        logging.info(f'Loading checkpoint: {args.resume}')
        load_checkpoint(model, args.resume)

    bit_config = bit_config_dict[bit_config_key]
    logging.info(f'Loading bit config: {bit_config}')
    set_bit_config(model, bit_config, args)

    logging.info('======================================================================')
    logging.info(model)
    logging.info('======================================================================')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    threshold = 1e-6
    max_patience = 15
    patience = 0
    best_epoch = 0
    best_acc1 = 0
    loss_record = list()
    acc_record = list()

    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        epoch_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        acc1 = validate(val_loader, model, criterion, args)

        loss_record.append(epoch_loss)
        acc_record.append(acc1)

        if np.abs(acc1-acc_record[-1]) < threshold:
            patience += 1
        else:
            patience = 0
        if patience == max_patience:
            logging.info(f'Patience {patience} reached. Exiting training.')
            break

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        logging.info(f'Best acc at epoch {epoch}: {best_acc1}')
        if is_best:
            # record the best epoch
            best_epoch = epoch

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'bit_config': bit_config
            }, is_best, save_path)

    try:
        filename = 'model_loss.json'
        with open(os.path.join(save_path, filename), 'w') as fp:
            json.dump(loss_record, fp)
    except:
        logging.error(f'Could not write training loss to {filename}')

    try:
        filename = f'model_{args.arch}_acc_{best_acc1:.4}.json'
        with open(os.path.join(save_path, filename), 'w') as fp:
            json.dump(acc_record, fp)
    except:
        logging.error(f'Could not write training accuracy to {filename}')

    try:
        # log model configuration and training info
        logging.info('======================================================================')
        logging.info(f'arch: {args.arch}')
        logging.info(f'best-acc: {best_acc1}')
        logging.info(f'best-epoch: {best_epoch+1}')
        logging.info(f'patience-readed: {patience==max_patience}/{max_patience}')
        logging.info(f'resume: {args.resume}')
        logging.info(f'bit-config-key: {bit_config_key}')
        logging.info(f'bit-config: {bit_config}')
        logging.info('======================================================================')
    except:
        logging.error('Could not log model configuration and training info.')


# python train.py --arch hawq_jettagger --lr 0.001 --batch-size 1024 --data data/ --save-path checkpoints/ --quant-scheme uniform8 --bias-bit 8 --quant-mode symmetric --epochs 15
if __name__ == '__main__':

    bit_configs = [
        'bit_config_hawq_jettagger_uniform4',
        'bit_config_hawq_jettagger_uniform6',
        'bit_config_hawq_jettagger_uniform8',
        'bit_config_hawq_jettagger_uniform12',
        'bit_config_hawq_jettagger_uniform16',
        'bit_config_hawq_jettagger_uniform24'
    ]

    # for config, bias in zip(config, config_bias):
    #     main(config)

    bit_config = "bit_config_" + args.arch + "_" + args.quant_scheme
    main(bit_config_key=bit_config)
