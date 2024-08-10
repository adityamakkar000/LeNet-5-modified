from train.train import train
import argparse

parser = argparse.ArgumentParser()


def list_of_ints(arg):
    return list(map(int, arg.split(",")))



parser.add_argument("-epochs", type=int, default=200)
parser.add_argument("-batch_size", type=int, default=256)
parser.add_argument("-batch_size_test", type=int, default=1000)
parser.add_argument("-lr", type=float, default=0.01)
parser.add_argument("-model", type=str, required=True)
parser.add_argument("-dims", type=list_of_ints, required=True)
parser.add_argument("-out_channels", type=list_of_ints)
parser.add_argument("-kernel_size", type=list_of_ints)

args = parser.parse_args()

epochs = args.epochs
batch_size_train = args.batch_size
batch_size_test = args.batch_size_test
learning_rate = args.lr
model = args.model
dims = args.dims


hyperparameters = {
    "epochs": epochs,
    "learning_rate": learning_rate,
    "dims": (*dims, 10),
    "cnn": {
        "out_channels": args.out_channels,
        "kernel_sizes": args.kernel_size
        }
}

# call train function
train(batch_size_train, batch_size_test, model, hyperparameters)
