import tensorflow as tf
from ops import *

d_bn1 = batch_norm(name='d_bn1')
d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')

def discriminator(image, reuse, num_filters=64, batch_size=64):
    # Das hier erlaubt das weiter trainineren der Matrizen mit den fake_images,
    # nachdem sie mit den originalen Bildern trainiert wurden.
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        h0 = lrelu(conv2d(image, num_filters, name='d_h0_conv'))
        print("discriminator h0")
        print(h0.shape)
        h1 = lrelu(d_bn1(conv2d(h0, num_filters*2, name='d_h1_conv')))
        print("discriminator h1")
        print(h1.shape)
        h2 = lrelu(d_bn2(conv2d(h1, num_filters*4, name='d_h2_conv')))
        print("discriminator h2")
        print(h2.shape)
        h3 = lrelu(d_bn3(conv2d(h2, num_filters*8, name='d_h3_conv')))
        print("discriminator h3")
        print(h3.shape)
        h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h3_lin')
        print("discriminator h4")
        print(h4.shape)
        return tf.nn.sigmoid(h4), h4
