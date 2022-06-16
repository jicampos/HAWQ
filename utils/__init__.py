from .data_utils import *
from .models.q_mobilenetv2 import *
from .models.q_inceptionv3 import *
from .models.q_resnet import *
from .models.q_jettagger import *
from .models.q_mnist import *

import shutil
import time
import logging
import warnings

from sklearn.metrics import accuracy_score

def load_checkpoint(model, filename):
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    for name, val in checkpoint['state_dict'].items():
        print(f'Loading {name}...')
        setattr(model, name, val)

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



def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    if args.fix_BN == True:
        model.eval()
    else:
        model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # if args.gpu is not None:
        #     images = images.cuda(args.gpu, non_blocking=True)
        # target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images.float())
        loss = criterion(output, target.float())

        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target.float(), topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        # top1.update(acc1[0], images.size(0))
        # top5.update(acc5[0], images.size(0))

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


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    freeze_model(model)
    model.eval()

    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # if args.gpu is not None:
            #     images = images.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images.float())
            loss = criterion(output, target.float())

            # measure accuracy and record loss
            # acc1, acc5 = accuracy(output, target.float(), topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            # top1.update(acc1[0], images.size(0))
            # top5.update(acc5[0], images.size(0))

            _, preds = torch.max(output, 1)
            predlist = torch.cat([predlist, preds.view(-1).cpu()])
            lbllist = torch.cat([lbllist, torch.max(target, 1)[1].view(-1).cpu()])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        
        outputs = output.cpu()
        local_labels = target.cpu()
        predict_test = outputs.numpy()
        accuracy_value = accuracy_score(np.nan_to_num(lbllist.numpy()), np.nan_to_num(predlist.numpy()))
        top1.update(accuracy_value, 1)

        logging.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    torch.save({'convbn_scaling_factor': {k: v for k, v in model.state_dict().items() if 'convbn_scaling_factor' in k},
                'fc_scaling_factor': {k: v for k, v in model.state_dict().items() if 'fc_scaling_factor' in k},
                'weight_integer': {k: v for k, v in model.state_dict().items() if 'weight_integer' in k},
                'bias_integer': {k: v for k, v in model.state_dict().items() if 'bias_integer' in k},
                'act_scaling_factor': {k: v for k, v in model.state_dict().items() if 'act_scaling_factor' in k},
                }, args.save_path + 'quantized_checkpoint.pth.tar')

    unfreeze_model(model)

    return top1.avg


def save_checkpoint(state, is_best, filename=None):
    torch.save(state, filename + 'checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(filename + 'checkpoint.pth.tar', filename + 'model_best.pth.tar')


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res