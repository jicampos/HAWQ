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
                                    batch_size=32,
                                    path='/data1/jcampos/HAWQ-main/data/train',
                                    for_inception=False,
                                    data_percentage=1)
val_loader = utils.getTestData(dataset='hlc_jets',
                                    batch_size=32,
                                    path='/data1/jcampos/HAWQ-main/data/val',
                                    data_percentage=0.1)


def main(config=None, bias=None):

    now = datetime.now() # current date and time
    date_time = now.strftime('%m%d%Y_%H%M%S')

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if not os.path.exists(os.path.join(args.save_path, date_time)):
        os.makedirs(os.path.join(args.save_path, date_time))

    save_path = os.path.join(args.save_path, date_time) + '/'

    print('----------------------------------------------------------')
    print(f'Saving to: {save_path}')
    print('----------------------------------------------------------')

    print(f'Loading {args.arch}...')
    model = quantize_arch_dict[args.arch]()

    if args.resume:
        load_checkpoint(model, args.resume)

    if config is None:
        config = "bit_config_" + args.arch + "_" + args.quant_scheme
    if bias is not None:
        args.bias_bit = bias

    
    bit_config = bit_config_dict[config]

    set_bit_config(model, bit_config, args)
    print(model)

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
            }, is_best, save_path)
    
    filename = 'model_loss_{}.json'.format(args.arch)
    with open(os.path.join(save_path, filename), 'w') as fp:
        json.dump(loss_record, fp)

    filename = 'model_{}_acc_{}.json'.format(args.arch, best_acc1)
    with open(os.path.join(save_path, filename), 'w') as fp:
        json.dump(acc_record, fp)
    
    f = open(os.path.join(save_path, "log.txt"), "x")
    f.write(f'arch: {args.arch}\n')
    f.write(f'bit-config: {config}\n')
    f.write(f'bias-bit: {args.bias_bit}\n')
    f.write(f'resume: {args.resume}\n')
    f.write(f'best-epoch: {best_epoch}\n')
    f.write(f'best-acc: {best_acc1}\n')
    f.close()


# python train.py --arch hawq_jettagger --lr 0.001 --batch-size 1024 --data data/ --save-path checkpoints/ --quant-scheme uniform8 --bias-bit 8 --quant-mode symmetric --epochs 15
# python train.py -a jettagger --epochs  --lr 0.001 --batch-size 1024 --data data/ --critoptoverride --save-path checkpoints/ --data-percentage 0.75  --checkpoint-iter -1 --quant-scheme uniform8 

if __name__ == '__main__':
    configs = [
        'bit_config_hawq_jettagger_uniform4',
        'bit_config_hawq_jettagger_uniform8',
        'bit_config_hawq_jettagger_uniform12',
        'bit_config_hawq_jettagger_uniform16',
        'bit_config_hawq_jettagger_uniform24'
    ]

    config_bias = [
        4,
        8,
        12, 
        16, 
        24
    ]
    
    # for config, bias in zip(config, config_bias):
    #     main(config, bias)

    main()
