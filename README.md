# multiview-generation-cnn

This repository contains the code for training a multiview generation CNN. The network takes in an image of an object and 
viewing angle as input, and generates an image of that object from that angle.
The approach closely follows the ECCV 2016 paper
[Multi-view 3D Models from Single Images with a Convolutional Network](https://arxiv.org/abs/1511.06702).
The code has been tested with TensorFlow version 1.3.

For training the network, run:\
`python train.py --exp <experiment name> --gpu <gpu id> --data_dir <data directory>`\

Sample results after training are provided in `samples/`.
