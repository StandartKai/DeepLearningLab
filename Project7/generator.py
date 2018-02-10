import tensorflow as tf

from ops import *

def generator(z, batch_size, z_dim=100, output_dim=[300,300], gf_dim=64, gfc_dim=1024, c_dim=3):
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

    # z = tf.placeholder(tf.float32, [None, z_dim], name='z')


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

        # now reshape it
        h0 = tf.reshape(z_, [-1, s_h16, s_w16, gf_dim * 8])

        # use normaliziation and relu.
        h0 = tf.nn.relu(g_bn0(h0))
        h0 = tf.layers.dropout(inputs=h0, rate=0.5)

        h1, h1_w, h1_b = deconv2d(h0, [batch_size, s_h8, s_w8, gf_dim * 4],
                                    name='g_h1', with_w=True)
        h1 = tf.nn.relu(g_bn1(h1))
        h1 = tf.layers.dropout(inputs=h1, rate=0.5)

        h2, h2_w, h2_b = deconv2d(h1, [batch_size, s_h4, s_w4, gf_dim * 2],
                                    name='g_h2', with_w=True)
        h2 = tf.nn.relu(g_bn2(h2))
        h2 = tf.layers.dropout(inputs=h2, rate=0.5)

        h3, h3_w, h3_b = deconv2d(h2, [batch_size, s_h2, s_w2, gf_dim * 1],
                                    name='g_h3', with_w=True)
        h3 = tf.nn.relu(g_bn3(h3))
        h3 = tf.layers.dropout(inputs=h3, rate=0.5)

        h4, h4_w, h4_b = deconv2d(h3, [batch_size, s_h, s_w, c_dim],
                                    name='g_h4', with_w=True)
        return tf.layers.dropout(inputs=tf.nn.tanh(h4), rate=0.5), h2
