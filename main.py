from train.train import train
import argparse

parser = argparse.ArgumentParser()


def list_of_ints(arg):
    return list(map(int, arg.split(",")))


parser.add_argument("-epochs", type=int, default=200)
parser.add_argument("-batch_size_train", type=int, default=256)
parser.add_argument("-batch_size_test", type=int, default=1000)
parser.add_argument("-lr", type=float, default=0.01)
parser.add_argument("-model", type=str, required=True)
parser.add_argument("-hidden_sizes", type=list_of_ints, required=True)
args = parser.parse_args()

epochs = args.epochs
batch_size_train = args.batch_size_train
batch_size_test = args.batch_size_test
learning_rate = args.lr
model = args.model
dims = args.hidden_sizes


hyperparameters = {
    "epochs": epochs,
    "learning_rate": learning_rate,
    "dims": (*dims, 10),
}

# call train function
train(batch_size_train, batch_size_test, model, hyperparameters)
