import numpy as np
from models.mlp import nMLP


class CNN:
    def __init__(
        self,
        stride,
        pool_shape,
        kernel_size,
        in_channels,
        out_channels,
        learning_rate,
        MLP_params,
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
                    kernel_size[index][1],
                ),
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
                kernel_y,
            ) = (
                image.shape[0],
                kernel.shape[0],
                image.shape[1],
                image.shape[2],
                image.shape[3],
                kernel.shape[2],
                kernel.shape[3],
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
                    image_y + padding_y,
                )
            )
            image_new[
                :,
                :,
                :,
                int(padding_x / 2) : image_x + int(padding_x / 2),
                int(padding_y / 2) : image_y + int(padding_y / 2),
            ] = np.repeat(np.expand_dims(image, axis=1), [out_channels], axis=1)

            conv_x, conv_y = (
                int((image_x + padding_x - kernel_x) / stride) + 1,
                int((image_y + padding_y - kernel_y) / stride) + 1,
            )

            ans = np.zeros((batch_size, out_channels, in_channels, conv_x, conv_y))

            for x in range(conv_x):
                for y in range(conv_y):
                    current_conv = image_new[
                        :,
                        :,
                        :,
                        x * stride : x * stride + kernel_x,
                        y * stride : y * stride + kernel_y,
                    ]

                    ans[:, :, :, x, y] += np.sum(
                        (kernel * current_conv).reshape(
                            batch_size, out_channels, in_channels, -1
                        ),
                        axis=-1,
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
                kernel_shape[1],
            )

            kernel = 0.25 * np.ones((in_channels, kernel_x, kernel_y))

            conv_x, conv_y = (
                int((image_x - kernel_x) / kernel_x) + 1,
                int((image_y - kernel_y) / kernel_y) + 1,
            )

            ans = np.zeros((batch_size, in_channels, conv_x, conv_y))

            for x in range(conv_x):
                for y in range(conv_y):
                    current_conv = image[
                        :,
                        :,
                        x * kernel_x : (x + 1) * kernel_x,
                        y * kernel_y : (y + 1) * kernel_y,
                    ]
                    ans[:, :, x, y] = np.sum(
                        (current_conv * kernel).reshape(batch_size, in_channels, -1),
                        axis=-1,
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
                kernel_y,
            ) = (
                image.shape[0],
                kernel.shape[0],
                kernel.shape[1],
                image.shape[2],
                image.shape[3],
                kernel.shape[2],
                kernel.shape[3],
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
                    image_y + padding_y,
                )
            )
            image_new[
                :,
                :,
                :,
                int(padding_x / 2) : image_x + int(padding_x / 2),
                int(padding_y / 2) : image_y + int(padding_y / 2),
            ] = image

            conv_x, conv_y = (
                int((image_x + padding_x - kernel_x) / stride) + 1,
                int((image_y + padding_y - kernel_y) / stride) + 1,
            )

            for x in range(conv_x):
                for y in range(conv_y):
                    current_conv = image_new[
                        :,
                        :,
                        :,
                        x * stride : x * stride + kernel_x,
                        y * stride : y * stride + kernel_y,
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
                kernel_y,
            ) = (
                dConv.shape[0],
                kernel.shape[0],
                kernel.shape[1],
                dConv.shape[2],
                dConv.shape[3],
                kernel.shape[2],
                kernel.shape[3],
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
                    image_y + padding_y,
                )
            )
            dConv = np.repeat(np.expand_dims(dConv, axis=2), [in_channels], axis=2)

            conv_x, conv_y = (
                int((image_x + padding_x - kernel_x) / stride) + 1,
                int((image_y + padding_y - kernel_y) / stride) + 1,
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
                        y * stride : y * stride + kernel_y,
                    ] = dCurrentConv

            dImageWithoutPadding = dImage[
                :,
                :,
                :,
                int(padding_x / 2) : image_x + int(padding_x / 2),
                int(padding_y / 2) : image_y + int(padding_y / 2),
            ]
            return np.sum(dImageWithoutPadding, axis=1)

        def inverse_pool(image, kernel_shape):
            batch_size, in_channels, image_x, image_y, kernel_x, kernel_y = (
                image.shape[0],
                image.shape[1],
                image.shape[2],
                image.shape[3],
                kernel_shape[0],
                kernel_shape[1],
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
                        y * kernel_y : (y + 1) * kernel_y,
                    ] = (
                        0.25 * current_tile
                    )

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
                axis=(1, 2),
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
