import tensorflow as tf

from ops import *

def generator(sess, z_dim, output_dim=[300,300], gf_dim=64, gfc_dim, c_dim):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    g_bn0 = batch_norm(name='g_bn0')
    g_bn1 = batch_norm(name='g_bn1')
    g_bn2 = batch_norm(name='g_bn2')
    g_bn3 = batch_norm(name='g_bn3')

    output_height = output_dim[0]
    output_width = output_dim[1]

    z = tf.placeholder(tf.float32, [None, z_dim], name='z')
    summary_z = tf.histogram_summary('z', z)

    with tf.variable_scope('generator') as scope:
        # Calculate the sizes for the transposed convolutional layer
        s_h, s_w = output_height, output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project z
        z_, h0_w, h0_b = linear(
            z, gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)
