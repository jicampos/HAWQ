import os
import logging
import json
from datetime import datetime 

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import utils 
from args import *
from utils import *
from bit_config import *

from utils import train
from utils import validate
from utils import save_checkpoint
from utils import adjust_learning_rate

quantize_arch_dict = {'jettagger': utils.models.q_jettagger.jettagger_model,
                      'hawq_jettagger': utils.models.q_jettagger.q_jettagger_model,
                      'brevitas_jettagger': None} #utils.models.q_jettagger.BrevitasJetTagger}

date_time = datetime.now().strftime('%Y%m%d_%H%M%S')

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if not os.path.exists(os.path.join(args.save_path, date_time)):
    os.makedirs(os.path.join(args.save_path, date_time))

args.save_path = os.path.join(args.save_path, date_time) + '/'

print('----------------------------------------------------------')
print(f'Saving to: {args.save_path}')
print('----------------------------------------------------------')


def main(config=None):

    train_loader = utils.getTrainData(dataset='hlc_jets',
                                    batch_size=32,
                                    path='/data1/jcampos/HAWQ-main/data/train',
                                    for_inception=False,
                                    data_percentage=0.1)
    val_loader = utils.getTestData(dataset='hlc_jets',
                                    batch_size=32,
                                    path='/data1/jcampos/HAWQ-main/data/val',
                                    data_percentage=0.1)

    print(f'Loading {args.arch}...')
    model = quantize_arch_dict[args.arch]()

    if args.resume:
        load_checkpoint(model, args.resume)

    if config is None:
        bit_config = bit_config_dict["bit_config_" + args.arch + "_" + args.quant_scheme]
    else:
        bit_config = bit_config_dict[config]

    set_bit_config(model, bit_config, args)
    print(model); return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

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
            }, is_best, args.save_path)
    
    filename = 'model_loss_{}.json'.format(args.arch)
    with open(os.path.join(args.save_path, filename), 'w') as fp:
        json.dump(loss_record, fp)

    filename = 'model_{}_acc_{}.json'.format(args.arch, best_acc1)
    with open(os.path.join(args.save_path, filename), 'w') as fp:
        json.dump(acc_record, fp)


# python train.py -a jettagger --epochs  --lr 0.001 --batch-size 1024 --data data/ --critoptoverride --save-path checkpoints/ --data-percentage 0.75  --checkpoint-iter -1 --quant-scheme uniform8 
if __name__ == '__main__':
    configs = [
        'bit_config_hawq_jettagger_uniform8',
        'bit_config_hawq_jettagger_uniform12',
        'bit_config_hawq_jettagger_uniform16',
        'bit_config_hawq_jettagger_uniform24'
    ]

    if args.bit_configs:
        for config in configs:
            main(config)
    else:
        main()
