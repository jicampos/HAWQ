import os
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import utils
from args import args
from bit_config import bit_config_dict
from train_utils import (
    get_quant_scheme,
    reset_logging,
    load_checkpoint,
    set_bit_config,
    adjust_learning_rate,
    train,
    validate,
    save_checkpoint,
    log_training,
)


quantize_arch_dict = {
    "jettagger": utils.models.q_jettagger.jettagger_model,
    "hawq_jettagger": utils.models.q_jettagger.q_jettagger_model,
}

now = datetime.now()  # current date and time
DATE_TIME = now.strftime("%m%d%Y_%H%M%S")


def main(bit_config_key, train_loader, val_loader):

    now = datetime.now()  # current date and time
    date_time = now.strftime("%m%d%Y_%H%M%S")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    quant_scheme = get_quant_scheme(bit_config_key)
    if not os.path.exists(os.path.join(args.save_path, quant_scheme, date_time)):
        os.makedirs(os.path.join(args.save_path, quant_scheme, date_time))

    save_path = os.path.join(args.save_path, quant_scheme, date_time) + "/"
    reset_logging()
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        filemode="w",
        datefmt="%d-%b-%y %H:%M:%S",
        filename=save_path + "training.log",
        encoding="utf-8",
        level=logging.DEBUG,
    )
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(args)

    model = quantize_arch_dict[args.arch](
        model=None,
        batchnorm=args.batch_norm,
        silu=args.silu,
        gelu=args.gelu,
        dense_out=args.dense_out,
    )
    teacher = None

    if args.resume:
        logging.info(f"Loading checkpoint: {args.resume}")
        load_checkpoint(model, args.resume, args)

    if args.distill_method == "KD_naive" or args.dc:
        teacher = quantize_arch_dict["hawq_jettagger"](
            model=None,
            batchnorm=args.batch_norm,
            silu=args.silu,
            gelu=args.gelu,
            dense_out=args.dense_out,
        )
        if args.teacher_checkpoint:
            teacher_checkpoint = (
                f"checkpoints/{args.teacher_checkpoint}/model_best.pth.tar"
            )
            logging.info(f"Loading fp32 weights from: {teacher_checkpoint}")
            # load pretrained weights to model
            load_checkpoint(teacher, teacher_checkpoint, args)
        else:
            logging.info(f"Using bit-config for teacher: {args.teacher_quant_scheme}")
            bit_config = bit_config_dict[
                f"bit_config_hawq_jettagger_{args.teacher_quant_scheme}"
            ]
            set_bit_config(teacher, bit_config, args)

    bit_config = bit_config_dict[bit_config_key]
    if args.arch == "hawq_jettagger":
        if args.train_scheme == "direct" or args.train_scheme == "gradual":
            # create non-quantized model
            fp32_model = quantize_arch_dict["jettagger"](model=model)
            fp32_pretrained = "checkpoints/fp32/06252022_134937/model_best.pth.tar"
            logging.info(f"Loading fp32 weights from: {fp32_pretrained}")
            # load pretrained weights to model
            load_checkpoint(fp32_model, fp32_pretrained, args)
            # quantize model
            model = quantize_arch_dict["hawq_jettagger"](model=fp32_model)
            # change train scheme if gradual
            args.train_scheme = (
                "gradual-mod" if args.train_scheme == "gradual" else args.train_scheme
            )
        logging.info(f"Loading bit config: {bit_config}")
        set_bit_config(model, bit_config, args)

    logging.info(
        "======================================================================"
    )
    logging.info(model)
    logging.info(
        "======================================================================"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    best_epoch = 0
    best_acc1 = 0
    loss_record = list()
    acc_record = list()

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        epoch_loss = train(
            train_loader, model, criterion, optimizer, epoch, args, teacher
        )
        acc1 = validate(val_loader, model, criterion, args)

        loss_record.append(epoch_loss)
        acc_record.append(acc1)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        logging.info(f"Best acc at epoch {epoch}: {best_acc1}")
        if is_best:
            # record the best epoch
            best_epoch = epoch

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                    "bit_config_key": bit_config_key,
                    "bit_config": bit_config,
                },
                is_best,
                save_path,
            )

    best_acc = f"{quant_scheme}/{date_time}: {best_acc1}\n"
    log_training(best_acc, f"{quant_scheme}/best.txt", args.save_path)
    log_training(best_acc, f"{DATE_TIME}.txt", args.save_path)
    log_training(loss_record, "model_loss.json", save_path)
    log_training(acc_record, "model_acc.json", save_path)

    try:
        # log model configuration and training info
        logging.info(
            "======================================================================"
        )
        logging.info(f"arch: {args.arch}")
        logging.info(f"best-acc: {best_acc1}")
        logging.info(f"best-epoch: {best_epoch+1}/{args.epochs}")
        logging.info(f"resume: {args.resume}")
        logging.info(f"train-scheme: {args.train_scheme}")
        logging.info(f"bit-config-key: {bit_config_key}")
        logging.info(f"bit-config: {bit_config}")
        logging.info(
            "======================================================================"
        )
    except:
        logging.error("Could not log model configuration and training info.")


# python train.py --arch hawq_jettagger --train-scheme random --epochs 10 --batch-norm --lr 0.001 --distill-method KD_naive --teacher-checkpoint uniform6/07012022_223011 --distill-alpha 0.1
if __name__ == "__main__":

    train_loader = utils.getTrainData(
        dataset="hlc_jets",
        batch_size=args.batch_size,
        path="/Users/jcampos/Documents/datasets/lhc_jets/train",
        for_inception=False,
        data_percentage=1,
    )
    val_loader = utils.getTestData(
        dataset="hlc_jets",
        batch_size=args.batch_size,
        path="/data1/jcampos/HAWQ-main/data/val",
        data_percentage=1,
    )

    bit_configs = ["bit_config_hawq_jettagger_uniform6"]

    for config in bit_configs:
        main(config, train_loader, val_loader)
