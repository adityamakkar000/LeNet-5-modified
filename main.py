import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import time

'n layer MLP with tanh nonlinearity'


class nMLP:
    def __init__(self, learning_rate, *dims):
        self.numLayers = len(dims) - 1  # numLayers - 1
        self.layers = [i for i in range(self.numLayers + 1)]  # numLayers
        self.dLayers = [i for i in range(self.numLayers)]  # (numLayers - 1)
        self.layerWeights = [
            np.random.randn(dims[i + 1], dims[i]) for i in range(0, self.numLayers)
        ]  # (numLayers - 1)
        self.layerBias = [
            np.random.randn(dims[i], 1) for i in range(1, self.numLayers + 1)
        ]  # (numLayers - 1)

        self.lr = learning_rate

    def get_total_params(self):
        total = np.sum(
            np.array([layer.shape[0] * layer.shape[1] for layer in self.layerWeights])
        )
        return total

    def __call__(self, X):
        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        self.layers[0] = X
        for l in range(0, self.numLayers - 1):
            self.layers[l + 1] = np.tanh(
                self.layers[l] @ self.layerWeights[l].T + self.layerBias[l].T
            )
        self.layers[self.numLayers] = softmax(
            self.layers[self.numLayers - 1] @ self.layerWeights[self.numLayers - 1].T
            + self.layerBias[self.numLayers - 1].T
        )

        self.output = self.layers[self.numLayers]

        return self.output

    def backward(self, y_true, print_loss=False, train=True):
        def CCE(predictions, y_true):
            # assuming y_true is in the form of 1 hot embeddings
            loss = -np.sum(y_true * np.log(predictions))
            return loss

        def MSE(predictions, y_true):
            loss = (
                0.5
                * (1 / predictions.shape[0])
                * np.sum(np.sum((predictions - y_true) ** 2, axis=-1), axis=0)
            )
            return loss

        loss = CCE(self.output, y_true)  # scalar 1,

        if print_loss:
            statment = f"Loss {loss}"
            print(statment)

        if train == False:
            return loss

        batch_size = self.output.shape[0]
        dW = [i for i in range(len(self.layerWeights))]
        dB = [i for i in range(len(self.layerBias))]

        # self.dOuput = (1/batch_size) * (y_true * (self.output ** -1)) # batch_size x ouput_dim
        # self.dLayers[self.numLayers - 1] = self.dOuput * (1 - (self.output)**2) # batch_size x output_dim

        self.dLayers[self.numLayers - 1] = self.output - y_true

        for l in range(self.numLayers - 2, -1, -1):
            self.dLayers[l] = (self.dLayers[l + 1] @ self.layerWeights[l + 1]) * (
                1 - (self.layers[l + 1]) ** 2
            )
        self.prev = self.dLayers[0] @ self.layerWeights[0] * (1 - (self.layers[0]) ** 2)

        for l in range(0, self.numLayers):
            dW[l] = (self.dLayers[l].T) @ self.layers[l]
            dB[l] = np.sum(self.dLayers[l].T, axis=-1, keepdims=True)

            self.layerWeights[l] -= self.lr * dW[l]
            self.layerBias[l] -= self.lr * dB[l]

        return loss


class CNN:
    def __init__(
        self,
        stride,
        pool_shape,
        kernel_size,
        in_channels,
        out_channels,
        learning_rate,
        MLP_params
    ):
        self.lr = learning_rate
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels + out_channels[:-1]
        self.out_channels = out_channels

        self.pool_shape = pool_shape

        self.kernels = [
            np.random.normal(
                0,
                np.sqrt(2 / kernel_size[index][0] * kernel_size[index][1]),
                (
                    oc,
                    self.in_channels[index],
                    kernel_size[index][0],
                    kernel_size[index][1]
                )
            )
            for index, oc in enumerate(self.out_channels)
        ]
        self.bias = [np.zeros((oc, 1, 1)) for oc in self.out_channels]

        self.MLP = nMLP(*MLP_params)

    def __call__(self, x):
        def conv(image, kernel, stride, bias=0):
            (
                batch_size,
                out_channels,
                in_channels,
                image_x,
                image_y,
                kernel_x,
                kernel_y
            ) = (
                image.shape[0],
                kernel.shape[0],
                image.shape[1],
                image.shape[2],
                image.shape[3],
                kernel.shape[2],
                kernel.shape[3]
            )

            padding_x = kernel_x - 1
            padding_y = kernel_y - 1

            if padding_x % 2 == 1:
                padding_x += 1
            if padding_y % 2 == 1:
                padding_y += 1

            image_new = np.zeros(
                (
                    batch_size,
                    out_channels,
                    in_channels,
                    image_x + padding_x,
                    image_y + padding_y
                )
            )
            image_new[
                :,
                :,
                :,
                int(padding_x / 2) : image_x + int(padding_x / 2),
                int(padding_y / 2) : image_y + int(padding_y / 2)
            ] = np.repeat(np.expand_dims(image, axis=1), [out_channels], axis=1)

            conv_x, conv_y = (
                int((image_x + padding_x - kernel_x) / stride) + 1,
                int((image_y + padding_y - kernel_y) / stride) + 1
            )

            ans = np.zeros((batch_size, out_channels, in_channels, conv_x, conv_y))

            for x in range(conv_x):
                for y in range(conv_y):
                    current_conv = image_new[
                        :,
                        :,
                        :,
                        x * stride : x * stride + kernel_x,
                        y * stride : y * stride + kernel_y
                    ]

                    ans[:, :, :, x, y] += np.sum(
                        (kernel * current_conv).reshape(
                            batch_size, out_channels, in_channels, -1
                        ),
                        axis=-1
                    )

            ans = np.sum(ans, axis=2)  # sum along channels

            ans = ans + bias
            return ans

        def pool(image, kernel_shape):
            """Average pooling with kernel striding"""

            batch_size, in_channels, image_x, image_y, kernel_x, kernel_y = (
                image.shape[0],
                image.shape[1],
                image.shape[2],
                image.shape[3],
                kernel_shape[0],
                kernel_shape[1]
            )

            kernel = 0.25 * np.ones((in_channels, kernel_x, kernel_y))

            conv_x, conv_y = (
                int((image_x - kernel_x) / kernel_x) + 1,
                int((image_y - kernel_y) / kernel_y) + 1
            )

            ans = np.zeros((batch_size, in_channels, conv_x, conv_y))

            for x in range(conv_x):
                for y in range(conv_y):
                    current_conv = image[
                        :,
                        :,
                        x * kernel_x : (x + 1) * kernel_x,
                        y * kernel_y : (y + 1) * kernel_y
                    ]
                    ans[:, :, x, y] = np.sum(
                        (current_conv * kernel).reshape(batch_size, in_channels, -1),
                        axis=-1
                    )

            return ans

        # conv layers
        self.layers = []
        self.layers.append(x)

        # len(self.kernels) represents the number of conv layers
        for convLayer in range(len(self.kernels)):
            x = self.layers[convLayer]
            kernels = self.kernels[convLayer]
            biases = self.bias[convLayer]
            current_conv_layer = conv(x, kernels, self.stride, biases)
            current_conv_layer = current_conv_layer * (current_conv_layer > 0)
            current_conv_layer = pool(current_conv_layer, self.pool_shape[convLayer])
            self.layers.append(current_conv_layer)

        final_conv = self.layers[-1]
        final_conv = final_conv.reshape(final_conv.shape[0], -1)
        final_conv = self.MLP(final_conv)

        return final_conv

    def backward(self, y_true, train_loss=True):
        def inverse_conv(image, dConv, kernel, stride):
            dKernel = np.zeros((kernel.shape))

            (
                batch_size,
                out_channels,
                in_channels,
                image_x,
                image_y,
                kernel_x,
                kernel_y
            ) = (
                image.shape[0],
                kernel.shape[0],
                kernel.shape[1],
                image.shape[2],
                image.shape[3],
                kernel.shape[2],
                kernel.shape[3]
            )

            dConv = np.repeat(np.expand_dims(dConv, axis=2), [in_channels], axis=2)
            image = np.repeat(np.expand_dims(image, axis=1), [out_channels], axis=1)

            padding_x = kernel_x - 1
            padding_y = kernel_y - 1

            if padding_x % 2 == 1:
                padding_x += 1
            if padding_y % 2 == 1:
                padding_y += 1

            image_new = np.zeros(
                (
                    batch_size,
                    out_channels,
                    in_channels,
                    image_x + padding_x,
                    image_y + padding_y
                )
            )
            image_new[
                :,
                :,
                :,
                int(padding_x / 2) : image_x + int(padding_x / 2),
                int(padding_y / 2) : image_y + int(padding_y / 2)
            ] = image

            conv_x, conv_y = (
                int((image_x + padding_x - kernel_x) / stride) + 1,
                int((image_y + padding_y - kernel_y) / stride) + 1
            )

            for x in range(conv_x):
                for y in range(conv_y):
                    current_conv = image_new[
                        :,
                        :,
                        :,
                        x * stride : x * stride + kernel_x,
                        y * stride : y * stride + kernel_y
                    ]
                    dCurrent_conv = (
                        np.expand_dims(dConv[:, :, :, x, y], axis=(3, 4)) * current_conv
                    )
                    dKernel += np.sum(dCurrent_conv, axis=0)

            return dKernel

        def inverse_conv_image(dConv, kernel, stride):
            (
                batch_size,
                out_channels,
                in_channels,
                image_x,
                image_y,
                kernel_x,
                kernel_y
            ) = (
                dConv.shape[0],
                kernel.shape[0],
                kernel.shape[1],
                dConv.shape[2],
                dConv.shape[3],
                kernel.shape[2],
                kernel.shape[3]
            )

            padding_x = kernel_x - 1
            padding_y = kernel_y - 1

            if padding_x % 2 == 1:
                padding_x += 1
            if padding_y % 2 == 1:
                padding_y += 1

            dImage = np.zeros(
                (
                    batch_size,
                    out_channels,
                    in_channels,
                    image_x + padding_x,
                    image_y + padding_y
                )
            )
            dConv = np.repeat(np.expand_dims(dConv, axis=2), [in_channels], axis=2)

            conv_x, conv_y = (
                int((image_x + padding_x - kernel_x) / stride) + 1,
                int((image_y + padding_y - kernel_y) / stride) + 1
            )

            for x in range(conv_x):
                for y in range(conv_y):
                    dCurrentTile = np.tile(
                        dConv[:, :, :, x, y], (kernel_x, kernel_y)
                    ).reshape(batch_size, out_channels, in_channels, kernel_x, kernel_y)
                    dCurrentConv = dCurrentTile * kernel
                    dImage[
                        :,
                        :,
                        :,
                        x * stride : x * stride + kernel_x,
                        y * stride : y * stride + kernel_y
                    ] = dCurrentConv

            dImageWithoutPadding = dImage[
                :,
                :,
                :,
                int(padding_x / 2) : image_x + int(padding_x / 2),
                int(padding_y / 2) : image_y + int(padding_y / 2)
            ]
            return np.sum(dImageWithoutPadding, axis=1)

        def inverse_pool(image, kernel_shape):
            batch_size, in_channels, image_x, image_y, kernel_x, kernel_y = (
                image.shape[0],
                image.shape[1],
                image.shape[2],
                image.shape[3],
                kernel_shape[0],
                kernel_shape[1]
            )

            ans = np.zeros(
                (batch_size, in_channels, image_x * kernel_x, image_y * kernel_y)
            )

            for x in range(0, image_x, kernel_x):
                for y in range(0, image_y, kernel_y):
                    current_tile = np.tile(image[:, :, x, y], kernel_shape).reshape(
                        batch_size, in_channels, kernel_x, kernel_y
                    )
                    ans[
                        :,
                        :,
                        x * kernel_x : (x + 1) * kernel_x,
                        y * kernel_y : (y + 1) * kernel_y
                    ] = (0.25 * current_tile)

            return ans

        loss = self.MLP.backward(y_true, print_loss=False, train=train_loss)
        if train_loss == False:
            return loss

        dFlatten = self.MLP.prev
        batch_size = self.layers[-1].shape[0]
        dPool = dFlatten.reshape(self.layers[-1].shape)

        loss = self.MLP.backward(y_true, print_loss=False)
        dFlatten = self.MLP.prev
        batch_size = self.layers[-1].shape[0]
        dPool = dFlatten.reshape(self.layers[-1].shape)

        for i in range(len(self.layers) - 1, 0, -1):
            dRelu = inverse_pool(dPool, self.pool_shape[i - 1])
            current_layer = inverse_pool(self.layers[i], self.pool_shape[i - 1])
            dConv = dRelu * (1.0 * (current_layer > 0))

            dk = inverse_conv(
                self.layers[i - 1], dConv, self.kernels[i - 1], stride=self.stride
            )
            db = np.expand_dims(
                np.sum(np.sum(dConv, axis=0).reshape(dConv.shape[1], -1), axis=-1),
                axis=(1, 2)
            ) * np.ones(self.bias[i - 1].shape)

            self.kernels[i - 1] = self.kernels[i - 1] - self.lr * np.clip(dk, -1, 1)
            self.bias[i - 1] = self.bias[i - 1] - self.lr * np.clip(db, -1, 1)

            dPool = inverse_conv_image(dConv, self.kernels[i - 1], stride=self.stride)

        return loss

    def get_total_params(self):
        total_params = 0
        for k in self.kernels:
            current_shape = 1
            for i in k.shape:
                current_shape *= i
            total_params += current_shape

        for b in self.bias:
            current_shape = 1
            for i in b.shape:
                current_shape *= i
            total_params += current_shape

        total_params += self.MLP.get_total_params()

        return total_params


epochs = 10
batch_size_train = 256
batch_size_test = 1000
learning_rate = 0.01

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        root='./files',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
    ),
    batch_size=batch_size_train,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        root='./files',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
    ),
    batch_size=batch_size_test,
    shuffle=True
)


def one_hot(array):
    length = array.size
    ans = np.zeros((length, 10))
    ans[np.arange(length), array] = 1
    return ans


dims = (784, 256, 32, 10)

MLP = nMLP(0.01, *dims)
print('MLP total params: ', MLP.get_total_params())
loss_arr_mlp = []
val_loss_arr_mlp = []
for _ in range(epochs):
    start = time.time()
    for batch_idx, (example_data, example_target) in enumerate(train_loader):
        example_data = (
            example_data.squeeze(dim=1).numpy().reshape(example_target.shape[0], -1)
        )
        example_target = one_hot(example_target.numpy())
        output = MLP(example_data)
        currentloss = MLP.backward(example_target, False)
        loss_arr_mlp.append(currentloss)

    val_loss = []
    for batch_idx, (example_data, example_target) in enumerate(test_loader):
        example_data = (
            example_data.squeeze(dim=1).numpy().reshape(example_target.shape[0], -1)
        )
        example_target = one_hot(example_target.numpy())

        output = MLP(example_data)
        loss = MLP.backward(example_target, False, False)
        val_loss.append(loss)
    mean_loss = np.mean(val_loss)
    val_loss_arr_mlp.append(mean_loss)
    end = time.time() - start
    print(
        'epoch ',
        _,
        'loss ',
        str(round(mean_loss, 4)),
        ' time for epoch ',
        str(round(end, 2)),
        's'
    )


CNN_params = {
    'stride': 1,
    'pool_shape': [(2, 2), (2, 2)],
    'kernel_size': [(5, 5), (5, 5)],
    'in_channels': [1],
    'out_channels': [6, 16],
    'learning_rate': 0.001,
    'MLP_params': (0.001, *dims)
}


leNet = CNN(**CNN_params)
loss = 10000
loss_arr = []

print('CNN total params: ', leNet.get_total_params())
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
        'epoch ',
        _,
        'loss ',
        str(round(mean_loss, 4)),
        ' time for epoch',
        str(round(end, 2)),
        's'
    )


def plot_loss(loss_array):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_array) + 1), loss_array, marker='o')
    plt.title('Loss vs. Batch Number')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


plot_loss(loss_arr_mlp)
plot_loss(loss_arr_cnn)
plot_loss(val_loss_arr_mlp)
plot_loss(val_loss_arr_cnn)
