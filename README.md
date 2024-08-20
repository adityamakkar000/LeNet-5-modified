# LeNet-5-modified

A pure numpy implemenation of LeNet-5 (slightly modified). Forward and backward passes written in numpy only.

## Setup

Setup a virtual env and run
``` bash
pip install -r req.txt
```

## Run

To run use the main file alongside the args for the corresponding model

### MLP

To run an mlp use

```bash
python3 main.py
  -model mlp
  -dims 64,32 # the dims of the hidden layers
  -lr $lr
  -epochs $epochs
  -batch_size $batch_size
```

### CNN

To run a CNN use

```bash
python3 main.py
 -model cnn
 -dims 64,32 # the dims of the feed-forward network
 -out_channels 3,6 # out channels for the conv layers
 -kernel_size 5,5 # size of the squared kernels
 -lr $lr
 -epochs $epochs
 -batch_size $batch_size

```

## File Structure

In the ``` models ``` folder, are the CNN and MLP implementation. In the ```models\torchmodels.py``` are torch equivalent to verify the accuracy. In the ```train``` folder the traning loops are implemented in the ```train.py``` folder.
