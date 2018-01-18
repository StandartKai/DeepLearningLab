from discriminator import *
from generator import *

batch_size = 64

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# size of eaach picture: 28 x 28

print(len(mnist.train.next_batch(batch_size)[0][0]))
