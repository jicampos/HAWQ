"""
    Exporting HAWQ models in QONNX.
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score

from args import *
from train_utils import load_checkpoint

from utils import q_jettagger_model, q_mnist
from utils.export import ExportManager
from utils.export import model_info

from utils.jet_dataset import JetTaggingDataset


def print_tensor(x_tensor):
    if len(x_tensor) > 1:
        x_tensor = x_tensor[0]
    x_tensor = x_tensor.reshape(-1)
    print("[ ", end="")
    for x in x_tensor:
        print(f"{x:.2f} ", end="")
    print(" ]")


# python export_example.py --arch hawq_jettagger --load uniform6/06252022_152008
# ------------------------------------------------------------
if __name__ == "__main__":

    print(f"Loading {args.arch}...")

    if args.arch == "hawq_jettagger":
        x = torch.randn([1, 16])
        hawq_model = q_jettagger_model(
            model=None,
            dense_out=args.dense_out,
            quant_out=args.quant_out,
            batchnorm=args.batch_norm,
            silu=args.silu,
            gelu=args.gelu,
        )
    elif args.arch == "hawq_mnist":
        x = torch.randn([1, 1, 28, 28])
        hawq_model = q_mnist()

    # load checkpoint
    if args.load:
        quant_scheme, date_tag = args.load.split("/")
        filename = f"hls4ml_hawq2qonnx_jet_{quant_scheme}_{date_tag}.onnx"
        load_checkpoint(hawq_model, f"checkpoints/{args.load}/model_best.pth.tar", args)

    # load jet dataset
    sample_idx = 0
    num_samples = 500
    dataset = JetTaggingDataset("/Users/jcampos/Documents/datasets/lhc_jets/val")
    x, y = dataset[sample_idx : sample_idx + num_samples]
    x, y, = torch.tensor(
        x
    ), torch.tensor(y)

    # create an export manager
    export_manager = ExportManager(hawq_model)
    # run inference with export modules
    export_pred = export_manager.predict(x)
    # export to qonnx 
    export_manager.export(x, filename)

    hawq_model.eval()
    hawq_pred = hawq_model(x)

    if type(hawq_pred) == tuple:
        export_pred = export_pred[0].numpy()
        hawq_pred, hawq_layer_out = hawq_pred

    y_test = np.argmax(y.detach().numpy(), axis=1)
    hawq_pred = np.argmax(hawq_pred.detach().numpy(), axis=1)
    export_pred = np.argmax(export_pred, axis=1)

    print(f"HAWQ Accuracy: {accuracy_score(y_test, hawq_pred):.4}")
    count = np.sum(y_test == export_pred)
    print(f"Export Accuracy: {accuracy_score(y_test, export_pred):.4}")

    if args.dense_out or args.quant_out:
        out = "dense_out" if args.dense_out else "quant_out"

        print("Compare Output of Layers (orginal vs export equivalent):")
        for idx, layer in enumerate(model_info[f"{out}_export_mode"].keys()):
            print(
                "-----------------------------------------------------------------------------"
            )
            hawq_out = hawq_layer_out[idx].numpy()
            export_layer_out = model_info[f"{out}_export_mode"][layer].numpy()
            print(f"Layer: {layer}")
            print(f"MSE: {((hawq_out-export_layer_out)**2).sum()}")
            print("True Layer output:")
            print_tensor(hawq_out)
            print("Export output:")
            print_tensor(export_layer_out)
        print(
            "-----------------------------------------------------------------------------"
        )

        print("HAWQ and ExportWrapper Output")
        print(f"MSE: {((hawq_pred-export_pred)**2).sum()}")
        print(
            "-----------------------------------------------------------------------------"
        )
