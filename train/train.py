import time
from models.mlp import nMLP
from models.cnn import CNN
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


def one_hot(array):
    length = array.size
    ans = np.zeros((length, 10))
    ans[np.arange(length), array] = 1
    return ans


def plot_loss(loss_array):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_array) + 1), loss_array, marker="o")
    plt.title("Loss vs. Batch Number")
    plt.xlabel("Batch Number")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


def CNNTrain(train_loader, test_loader, learning_rate, epochs, dims, cnn):
    dims = (int((28/(2**(len(cnn["out_channels"]))))**2  * cnn["out_channels"][-1]), *dims)
    CNN_params = {
        "stride": 1,
        "pool_shape": [(2, 2) for i in cnn["out_channels"]],
        "kernel_size": [(i, i) for i in cnn["kernel_sizes"]],
        "in_channels": [1],
        "out_channels": [*cnn["out_channels"]],
        "learning_rate": learning_rate,
        "MLP_params": (learning_rate, *dims),
    }

    leNet = CNN(**CNN_params)
    loss = 10000

    print("CNN total params: ", leNet.get_total_params())
    loss_arr_cnn = []
    val_loss_arr_cnn = []
    for _ in range(epochs):
        start = time.time()
        for batch_idx, (example_data, example_target) in enumerate(train_loader):
            example_data = example_data.squeeze(dim=1).numpy()
            example_target = one_hot(example_target.numpy())
            example_data = np.expand_dims(example_data, axis=1)

            output = leNet(example_data)
            currentloss = leNet.backward(example_target)
            loss_arr_cnn.append(currentloss)

        val_loss = []
        for batch_idx, (example_data, example_target) in enumerate(test_loader):
            example_data = example_data.squeeze(dim=1).numpy()
            example_target = one_hot(example_target.numpy())
            example_data = np.expand_dims(example_data, axis=1)

            output = leNet(example_data)
            loss = leNet.backward(example_target, False)
            val_loss.append(loss)
        mean_loss = np.mean(val_loss)
        val_loss_arr_cnn.append(mean_loss)
        end = time.time() - start
        print(
            "epoch ",
            _,
            "loss ",
            str(round(mean_loss, 4)),
            " time for epoch",
            str(round(end, 2)),
            "s",
        )

    plot_loss(loss_arr_cnn)
    plot_loss(val_loss_arr_cnn)


def MLPTrain(train_loader, test_loader, learning_rate, epochs, dims, cnn):
    dims = (784, *dims)

    MLP = nMLP(learning_rate, *dims)
    print("MLP total params: ", MLP.get_total_params())

    loss_arr_mlp = []
    val_loss_arr_mlp = []

    for _ in range(epochs):
        start = time.time()
        for batch_idx, (example_data, example_target) in enumerate(train_loader):
            example_data = (
                example_data.numpy().reshape(example_target.shape[0], -1)
            )
            example_target = one_hot(example_target.numpy())
            output = MLP(example_data)
            currentloss = MLP.backward(example_target, False)
            loss_arr_mlp.append(currentloss)

        val_loss = []
        for batch_idx, (example_data, example_target) in enumerate(test_loader):
            example_data = (
                example_data.numpy().reshape(example_target.shape[0], -1)
            )
            example_target = one_hot(example_target.numpy())

            output = MLP(example_data)
            loss = MLP.backward(example_target, False, False)
            val_loss.append(loss)
        mean_loss = np.mean(val_loss)
        val_loss_arr_mlp.append(mean_loss)
        end = time.time() - start
        print(
            "epoch ",
            _,
            "loss ",
            str(round(mean_loss, 4)),
            " time for epoch ",
            str(round(end, 2)),
            "s",
        )

    plot_loss(loss_arr_mlp)
    plot_loss(val_loss_arr_mlp)


def train(batch_size_train, batch_size_test, model, args):

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root="./files",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size_train,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root="./files",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size_test,
        shuffle=True,
    )

    models = {
        "mlp": MLPTrain,
        "cnn": CNNTrain,
    }

    models[model](train_loader, test_loader, **args)
