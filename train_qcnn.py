import torch
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
from sklearn.metrics import accuracy_score

from utils import q_mnist, mnist
from utils.mnist_dataset import MNISTDataset


def load_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_data = MNISTDataset("./datasets/mnist", train=True, transform=transform)
    test_data = MNISTDataset("./datasets/mnist", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1024, shuffle=True, num_workers=1, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1024, shuffle=True, num_workers=1, pin_memory=True
    )

    return train_loader, test_loader


def load_model(quantized=True):
    if quantized:
        model = q_mnist()
    else:
        model = mnist()
    return model


def accuracy(y_true, y_pred):
    if type(y_true) == torch.Tensor:
        y_true = y_true.detach().numpy()
    if type(y_pred) == torch.Tensor:
        y_pred = y_pred.detach().numpy()
    if len(y_pred.shape) > 1:
        return accuracy_score(y_true, np.argmax(y_pred, axis=1))
    return accuracy_score(y_true, y_pred)


def train(model, train_loader, optimizer, epoch):
    dataset_size = len(train_loader) * 1024
    samples_passed = 0
    print_freq = dataset_size % 20

    model.train()
    for x_train, target in train_loader:
        output = model(x_train)

        # compute loss and accuracy for batch
        train_acc = accuracy(target, output)
        loss = F.nll_loss(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        samples_passed += len(x_train)
        if samples_passed % print_freq == 0:
            print(
                f"[{epoch+1}] [{samples_passed}/{dataset_size}] - acc {train_acc:.4f} loss {loss:.4f}"
            )
    return train_acc


def validate(model, test_loader, epoch):
    dataset_size = len(test_loader) * 1024
    samples_passed = 0
    print_freq = 1024
    pred_list = torch.zeros(0, dtype=torch.long, device="cpu")
    true_list = torch.zeros(0, dtype=torch.long, device="cpu")

    model.eval()
    with torch.no_grad():
        for x_train, target in test_loader:
            output = model(x_train)

            pred_list = torch.cat([pred_list, torch.max(output, 1)[1]])
            true_list = torch.cat([true_list, target])

            # compute loss and accuracy for batch
            val_acc = accuracy(target, output)
            loss = F.nll_loss(output, target)

        samples_passed += len(x_train)
        if samples_passed % print_freq == 0:
            print(
                f"[{epoch+1}] [{samples_passed}/{dataset_size}] - acc {val_acc:.4f} loss {loss:.4f}"
            )
    return accuracy(true_list, pred_list)


def save_checkpoint(epoch, best_acc, model, optimizer, filename):
    torch.save(
        {
            "epoch": epoch + 1,
            "best_acc": best_acc,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        filename,
    )


def main():
    max_epochs = 1
    best_acc = 0

    train_loader, val_loader = load_dataset()
    model = load_model(quantized=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(max_epochs):
        train_acc = train(model, train_loader, optimizer, epoch)
        val_acc = validate(model, val_loader, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                epoch,
                best_acc,
                model,
                optimizer,
                "checkpoints/cnn/qcnn_model_best.pth.tar",
            )

        print(
            "==========================================================================="
        )
        print(f"[{epoch+1}] train {train_acc:.4f} val {val_acc:.4f}")
        print(
            "==========================================================================="
        )
    return model


if __name__ == "__main__":
    model = main()

    from utils.export import ExportManager

    # create an export manager
    export_manager = ExportManager(model)
    # run model inference on hawq and export modules
    x = torch.randn([1, 28, 28])
    export_pred, hawq_pred = export_manager(x)
    # export the model to qonnx
    export_manager.export(x, "hawq2qonnx_cnn.onnx")
