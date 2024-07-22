
import numpy as np

"""n layer MLP with tanh nonlinearity"""

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

    def backward(self, y_true, print_loss=False):

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

        for l in range(0, self.numLayers):
            dW[l] = (self.dLayers[l].T) @ self.layers[l]
            dB[l] = np.sum(self.dLayers[l].T, axis=-1, keepdims=True)

            self.layerWeights[l] -= self.lr * dW[l]
            self.layerBias[l] -= self.lr * dB[l]



def conv(image, kernel, stride, bias=0):

    padding_x = kernel.shape[0] - 1
    padding_y = kernel.shape[1] - 1

    if padding_x % 2 == 1:
        padding_x += 1
    if padding_y % 2 == 1:
        padding_y += 1

    image_new = np.zeros((image.shape[0] + padding_x, image.shape[1] + padding_y))
    image_new[
        int(padding_x / 2) : image.shape[0] + int(padding_x / 2),
        int(padding_y / 2) : image.shape[1] + int(padding_y / 2),
    ] = image
    ans = (
        np.zeros(
            (
                int((image.shape[0] + padding_x - kernel.shape[0]) / stride) + 1,
                int((image.shape[1] + padding_y - kernel.shape[1]) / stride) + 1,
            )
        )
        + bias
    )

    for x in range(ans.shape[0]):
        for y in range(ans.shape[1]):
            current_conv = image_new[
                x * stride : x * stride + kernel.shape[0],
                y * stride : y * stride + kernel.shape[1],
            ]
            ans[x, y] += np.sum(kernel * current_conv)

    return ans


def pool(image, kernel_shape, stride):
    """Average pooling"""
    kernel = 0.25 * np.ones(kernel_shape)
    ans = np.zeros(
        (
            int((image.shape[0] - kernel.shape[0]) / stride) + 1,
            int((image.shape[1] - kernel.shape[1]) / stride) + 1,
        )
    )

    for x in range(0, ans.shape[0]):
        for y in range(0, ans.shape[1]):

            current_conv = image[
                x * stride : x * stride + kernel.shape[0],
                y * stride : y * stride + kernel.shape[1],
            ]
            ans[x, y] = np.sum(current_conv * kernel)
    return ans


class CNN:

    def __init__(
        self, stride, pool_shape, pool_stride, kernel_size, in_channels, out_channels, MLP_params
    ):

        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels + out_channels[:-1]
        self.out_channels = out_channels

        self.pool_shape = pool_shape
        self.pool_stride = pool_stride
        print(self.in_channels)

        self.kernels = [
            np.array(
                [
                    [
                        np.random.randn(kernel_size[index][0], kernel_size[index][1])
                        for i in range(self.in_channels[index])
                    ]
                    for _ in range(oc)
                ]
            )
            for index, oc in enumerate(self.out_channels)
        ]

        self.bias = [
            np.array(
                [
                    [
                        np.random.randn(
                            1,
                        )
                        for i in range(self.in_channels[index])
                    ]
                    for _ in range(oc)
                ]
            )
            for index, oc in enumerate(self.out_channels)
        ]

        self.MLP = nMLP(*MLP_params)

    def __call__(self, x):

        # conv layers
        for convLayer in range(len(self.kernels)):
            ans = []
            # out channels
            for kernels, biases in zip(self.kernels[convLayer], self.bias[convLayer]):

                # in chanells
                current_ans = np.sum(
                    np.array(
                        [
                            conv(x[index], k, self.stride, b)
                            for index, (k, b) in enumerate(zip(kernels, biases))
                        ]
                    ),
                    axis=0,
                )
                current_ans = np.tanh(current_ans)
                current_ans = pool(
                    current_ans, self.pool_shape[convLayer], self.pool_stride[convLayer]
                )

                ans.append(current_ans)
            x = ans

        ans = np.array(ans)
        print(ans.shape)
        ans = ans.reshape(-1)
        ans = self.MLP(ans)

        return ans

    def backward():

      #TODO
      return 0


import torch
import torchvision

epochs = 3
batch_size_train = 256
batch_size_test = 1000
learning_rate = 0.01

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

def one_hot(array):
  length = array.size
  ans = np.zeros((length, 10))
  ans[np.arange(length), array] = 1
  return ans

# train MLP
MLP = nMLP(learning_rate, 784, 256, 32, 10)
print(MLP.get_total_params())

for _ in range(epochs):

  for batch_idx, (example_data, example_target) in enumerate(train_loader):

    example_data = example_data.squeeze(dim=1).numpy().reshape(example_target.shape[0], -1)
    example_target = one_hot(example_target.numpy())
    output = MLP(example_data)
    MLP.backward(example_target, True if batch_idx % 100 == 0 else False)


# leNet

""" LeNet5 modified version


conv 5x5 stride 1 padding preserves dimension 6 channels
tanh
pool average 2x2 stride 2
conv 5x5 stride 1 padding preserves dimension 16 channels
tanh
pool average 2x2 stride 2
flatten

# MLP
Linear 784
tanh
Linear 120
tanh
Linear 84
tanh
Linear 10
Softmax

"""


dims = (784, 120,84, 10)
MLP_params = (learning_rate, *dims)
CNN_params = {
    "stride": 1,
    "pool_shape": [(2,2), (2,2)],
    "pool_stride": [2,2],
    "kernel_size": [(5,5), (5,5)],
    "in_channels": [1],
    "out_channels": [6,16],
    "MLP_params": MLP_params
              }


letNet = CNN(**CNN_params)