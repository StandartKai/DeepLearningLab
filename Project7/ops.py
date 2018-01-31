import tensorflow as tf
import math
import numpy as np
import h5py
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                  initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv


def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
    """
        Args:
            input_ = Input tensor
            output_shape = shape of the output tensor
            k_h = kernel height
            k_w = kernel width
            d_h, d_w = stride height and width
    """
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                      strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                      strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                     tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
    if with_w:
        return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
        return tf.matmul(input_, matrix) + bias


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                        decay=self.momentum,
                        updates_collections=None,
                        epsilon=self.epsilon,
                        scale=True,
                        is_training=train,
                        scope=self.name)

def loadDataFromMNIST(path=None):
    """ Loads the data from MNIST
    Example usage of MNIST data:
        batch = mnist.train.next_batch(<batch_size>)
        batch[0] --> image, batch[1] --> labels

        mnist.test.images
        mnist.test.labels
    """
    path = path if path is not None else '/tmp/tensorflow/mnist/input_data'
    return input_data.read_data_sets(path, one_hot=True)

def init_vars():
    try:
        tf.global_variables_initializer().run()
    except:
        tf.initialize_all_variables().run()

def saveNoiseAndImage(data, z_dim):
    print('### Saving images and noise vectors')
    h5f = h5py.File('data.h5', 'w')
    h5f.create_dataset('image_with_noise', data=data)
    h5f.attrs['z_dim'] = z_dim
    h5f.close()
    print('### Finished saving images and noise vectors')

def loadNoiseAndImage():
    print('### Loading saved images and noise vectors')
    h5f = h5py.File('data.h5', 'r')
    data = h5f['image_with_noise'][:]
    z_dim = h5f.attrs['z_dim']
    images, noise_vectors = data[:,:-z_dim], data[:,-z_dim:]
    h5f.close()
    print('### Finished loading images and noise vectors')
    return images, noise_vectors

def saveImage(image_data, height, width, name):
    image_data_reshaped = np.reshape(image_data, (height, width))
    plt.imshow(image_data_reshaped, vmin=0, vmax=1, cmap='gray')
    plt.savefig('./pictures/' + name + '.png')
