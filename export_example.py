"""
    Exporting HAWQ models in QONNX.
"""
import os
import sys
import logging
import argparse
import warnings
import numpy as np

import onnx
import onnxoptimizer
import qonnx

import torch
import torch.autograd
from torch.onnx import register_custom_op_symbolic
from torch._C import ListType, OptionalType

from args import *
from utils.export import *
from utils import q_jettagger_model, q_resnet18, q_mnist
from utils.quantization_utils.quant_modules import QuantAct, QuantDropout, QuantLinear, QuantBnConv2d
from utils.quantization_utils.quant_modules import QuantMaxPool2d, QuantAveragePool2d, QuantConv2d

from utils.JetTaggingDataset import JetTaggingDataset

# python export_example.py --arch hawq_jettagger --load uniform6/06252022_152008
# ------------------------------------------------------------
if __name__ == '__main__':

    print(f'Loading {args.arch}...\n')

    if args.arch == 'hawq_jettagger':
        filename = 'hawq2qonnx_jet.onnx'
        x = torch.randn([1, 16])
        hawq_model = q_jettagger_model(model=None, dense_out=args.dense_out, quant_out=args.quant_out,
                                        batchnorm=args.batch_norm, silu=args.silu, gelu=args.gelu)
    elif args.arch == 'hawq_mnist':
        filename = 'hawq2qonnx_conv.onnx'
        x = torch.randn([1, 1, 28, 28])
        hawq_model = q_mnist()

    print('Original layers:')
    print('-----------------------------------------------------------------------------')
    print(hawq_model)
    print('-----------------------------------------------------------------------------')

    if args.load:
        quant_scheme, date_tag = args.load.split('/')
        filename = f'fixed_v2_hawq2qonnx_jet_{quant_scheme}_{date_tag}.onnx'
        from train_utils import load_checkpoint
        load_checkpoint(hawq_model, f'checkpoints/{args.load}/model_best.pth.tar', args)

    model = ExportManager(hawq_model)

    print('Original layers:')
    print('-----------------------------------------------------------------------------')
    print(model)
    print('-----------------------------------------------------------------------------')

    model.export(x, filename=filename, save=True)

    dataset = JetTaggingDataset('/data1/jcampos/datasets/val')
    num_samples = 50000
    x, y = dataset[:num_samples]
    x, y = torch.tensor(x).reshape([-1, 16]), torch.tensor(y).reshape([-1, 5])
    print(x.shape, y.shape)
    e_pred = model(x)
    pred = hawq_model(x)

    if type(pred) == tuple:
        e_pred = e_pred[0]
        pred, true_out = pred

    y = np.argmax(y.detach().numpy(), axis=1)
    hawq_pred = np.argmax(pred.detach().numpy(), axis=1)
    export_pred = np.argmax(e_pred.detach().numpy(), axis=1)

    count = np.sum(hawq_pred != export_pred)
    print(f'model difference: {count/num_samples:.2}')
    count = np.sum(y == hawq_pred)
    print(f'HAWQ Accuracy: {count/num_samples:.2}')
    count = np.sum(y == export_pred)
    print(f'Export Accuracy: {count/num_samples:.2}')

    if args.dense_out or args.quant_out:
        out = 'dense_out' if args.dense_out else 'quant_out'

        print('Compare Output of Layers (orginal vs export equivalent):')
        for idx, layer in enumerate(model_info[f'{out}_export_mode'].keys()):
            print('-----------------------------------------------------------------------------')
            t_out = true_out[idx]
            layer_out = model_info[out][layer] 
            export_layer_out = model_info[f'{out}_export_mode'][layer] 
            print(f'Layer: {layer}')
            print(f'MSE (True+Export): {((t_out-export_layer_out)**2).sum()}')
            print(f'True Layer output: {t_out}')
            print(f'Layer output: {layer_out}')
            print(f'Export output: {export_layer_out.detach().numpy()}')
        print('-----------------------------------------------------------------------------')

        print('HAWQ and ExportWrapper Output')
        print('-----------------------------------------------------------------------------')
        print(f'MSE: {((pred.detach().numpy()-e_pred.detach().numpy())**2).sum()}')
        print('-----------------------------------------------------------------------------')
