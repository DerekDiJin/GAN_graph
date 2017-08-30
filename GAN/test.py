import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
from scipy import sparse

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

X, Y = mnist.train.next_batch(1)

print(X, Y)
print((X.shape, Y.shape))