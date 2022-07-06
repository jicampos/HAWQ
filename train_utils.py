import os 
import json
import time
import shutil
import logging
import warnings

import numpy as np
import torch.nn as nn 
from sklearn.metrics import accuracy_score

from args import args
from utils.quantization_utils.quant_modules import *


# ------------------------------------------------------------
# logging 
# ------------------------------------------------------------
def log_training(data, filename, save_path):
    try:
        path_to_file = os.path.join(save_path, filename)
        with open(path_to_file, 'a') as fp:
            if type(data) == str:
                fp.write(data)
            else:
                json.dump(data, fp)
    except:
        logging.error(f'Error while logging to file {path_to_file}')

def reset_logging():
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)

def get_quant_scheme(bit_config_key):
    return bit_config_key.split('bit_config_hawq_jettagger_')[1]

# ------------------------------------------------------------
# checkpoints
# ------------------------------------------------------------
def set_bit_config(model, bit_config, args):
    for name, module in model.named_modules():
        if name in bit_config.keys():
            setattr(module, 'quant_mode', args.quant_mode)
            setattr(module, 'bias_bit', args.bias_bit)
            setattr(module, 'quantize_bias', (args.bias_bit != 0))
            setattr(module, 'per_channel', args.channel_wise)
            setattr(module, 'act_percentile', args.act_percentile)
            setattr(module, 'act_range_momentum', args.act_range_momentum)
            setattr(module, 'weight_percentile', args.weight_percentile)
            setattr(module, 'fix_flag', False)
            setattr(module, 'fix_BN', args.fix_BN)
            setattr(module, 'fix_BN_threshold', args.fix_BN_threshold)
            setattr(module, 'training_BN_mode', args.fix_BN)
            setattr(module, 'checkpoint_iter_threshold', args.checkpoint_iter)
            setattr(module, 'save_path', args.save_path)
            setattr(module, 'fixed_point_quantization', args.fixed_point_quantization)

            if type(bit_config[name]) is tuple:
                bitwidth = bit_config[name][0]
                if bit_config[name][1] == 'hook':
                    module.register_forward_hook(hook_fn_forward)
                    global hook_keys
                    hook_keys.append(name)
            else:
                bitwidth = bit_config[name]
                if hasattr(module, 'bias'):
                    bias_bitwidth = bit_config[name+'_bias']

            if hasattr(module, 'activation_bit'):
                setattr(module, 'activation_bit', bitwidth)
                if bitwidth == 4:
                    setattr(module, 'quant_mode', 'asymmetric')
            else:
                setattr(module, 'weight_bit', bitwidth)
                if hasattr(module, 'bias'):
                    setattr(module, 'bias_bit', bias_bitwidth)

def load_checkpoint(model, filename, args):
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    if 'fp32' in filename:
        model.load_state_dict(checkpoint['state_dict'])
        model.train()
        return

    set_bit_config(model, checkpoint['bit_config'], args)
    checkpoint = checkpoint['state_dict']
    model_dict  = model.state_dict()
    modified_dict  = {}

    for key, value in checkpoint.items():
        if model_dict[key].shape != value.shape:
            print(f'mismatch: {key}: {value}')
            value = torch.tensor([value], dtype=torch.float64)
        modified_dict[key] = value
    model.load_state_dict(modified_dict, strict=False)
    model.train()

# ------------------------------------------------------------
# training
# ------------------------------------------------------------
def domain_discrepancy(source_feat, target_feat, loss_type='kl'):
    discrep_loss = 0
    if loss_type == 'kl':
        from torch.nn.functional import kl_div
        for target, source in zip(target_feat, source_feat):
            discrep_loss += kl_div(target.detach(), source.detach(), reduction='batchmean')
    elif loss_type == 'norm':
        for target, source in zip(target_feat, source_feat):
            difference = target.detach().reshape(-1,1)-source.detach().reshape(-1, 1)
            discrep_loss += torch.norm(difference)
    return discrep_loss * (1-args.distill_alpha)

def loss_kd(output, target, teacher_output, args):
    """
    Compute the knowledge-distillation (KD) loss given outputs and labels.
    "Hyperparameters": temperature and alpha
    The KL Divergence for PyTorch comparing the softmaxs of teacher and student.
    The KL Divergence expects the input tensor to be log probabilities.
    """
    alpha = args.distill_alpha
    T = args.temperature
    criterion = nn.BCELoss()
    KD_loss = F.kl_div(F.log_softmax(output / T, dim=1), F.softmax(teacher_output / T, dim=1), reduction='batchmean') * (alpha * T * T) + \
             criterion(output, target.float()) * (1. - alpha)

    return KD_loss

# ------------------------------------------------------------
def train(train_loader, model, criterion, optimizer, epoch, args, teacher=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracy = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, accuracy],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    if args.fix_BN == True:
        model.eval()
    else:
        model.train()
    lossdc = 0
    end = time.time()
    for i, (X_train, y_train) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(X_train.float())

        if args.dc:
            output, output_feats = output[0], output[1]
            _, teach_feats = teacher(X_train.float())
            lossdc = domain_discrepancy(output_feats, teach_feats, loss_type='kl')

        if args.distill_method == 'None':
            loss = criterion(output, y_train.float()) + lossdc
        elif args.distill_method == 'KD_naive':
            teacher_output = teacher(X_train.float())
            loss = loss_kd(output, y_train.float(), teacher_output, args) + lossdc
        else:
            raise Exception(f'Distill Method {args.distill_method} is not implemented')

        # measure accuracy and record loss
        losses.update(loss.item(), X_train.size(0))

        _, preds = torch.max(output, 1)
        tmp_pred = preds.view(-1).cpu()
        tmp_lbl = torch.max(y_train, 1)[1].view(-1).cpu()
        acc = accuracy_score(np.nan_to_num(tmp_lbl.numpy()), np.nan_to_num(tmp_pred.numpy()))
        # update progress meter 
        accuracy.update(acc, X_train.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
    return losses.avg

# def validate_accuracy(val_dataset, model)

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracy = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, accuracy],
        prefix='Test: ')

    # switch to evaluate mode
    freeze_model(model)
    model.eval()
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

    with torch.no_grad():
        end = time.time()
        for i, (X_val, y_val) in enumerate(val_loader):
            # compute output
            output = model(X_val.float())

            if args.dc:
                output, _ = output[0], output[1]

            loss = criterion(output, y_val.float())
            
            # measure accuracy and record loss
            losses.update(loss.item(), X_val.size(0))

            _, preds = torch.max(output, 1)
            tmp_pred = preds.view(-1).cpu()
            tmp_lbl = torch.max(y_val, 1)[1].view(-1).cpu()
            acc = accuracy_score(np.nan_to_num(tmp_lbl.numpy()), np.nan_to_num(tmp_pred.numpy()))
            # update progress meter 
            accuracy.update(acc, X_val.size(0))

            predlist = torch.cat([predlist, preds.view(-1).cpu()])
            lbllist = torch.cat([lbllist, torch.max(y_val, 1)[1].view(-1).cpu()])
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        acc = accuracy_score(np.nan_to_num(lbllist.numpy()), np.nan_to_num(predlist.numpy()))
        logging.info(' * Acc {accuracy:.3f}'.format(accuracy=acc))
        accuracy.update(acc, 1)
        logging.info(' * Acc {accuracy.avg:.3f}'.format(accuracy=accuracy))

    torch.save({'convbn_scaling_factor': {k: v for k, v in model.state_dict().items() if 'convbn_scaling_factor' in k},
                'fc_scaling_factor': {k: v for k, v in model.state_dict().items() if 'fc_scaling_factor' in k},
                'weight_integer': {k: v for k, v in model.state_dict().items() if 'weight_integer' in k},
                'bias_integer': {k: v for k, v in model.state_dict().items() if 'bias_integer' in k},
                'act_scaling_factor': {k: v for k, v in model.state_dict().items() if 'act_scaling_factor' in k},
                }, args.save_path + 'quantized_checkpoint.pth.tar')

    unfreeze_model(model)
    return accuracy.avg

def save_checkpoint(state, is_best, filename=None):
    torch.save(state, filename + 'checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(filename + 'checkpoint.pth.tar', filename + 'model_best.pth.tar')

# ------------------------------------------------------------
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    print('lr = ', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
