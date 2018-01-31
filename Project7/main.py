from discriminator import *
from generator import *
from ops import loadDataFromMNIST

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from six.moves import xrange

# size of eaach picture: 28 x 28
def main(sess, restore=True):

    BATCH_SIZE = 64
    NUM_EPOCHES = 10000
    INPUT_HEIGHT = 28
    INPUT_WIDTH = 28
    # Color dimension: e.g 1 for grayscale and 3 for RGB
    C_DIM = 1
    # number of dimension of a label
    Y_DIM = 10
    # number of elements in generator conv2d_transpose
    Z_DIM = 100

    # optimizer variables
    LEARNING_RATE = 0.002
    BETA_1 = 0.5

    mnist = loadDataFromMNIST()


    #x = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_WIDTH*INPUT_HEIGHT])
    y = tf.placeholder(tf.float32, [BATCH_SIZE, Y_DIM], name='y')

    inputs = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGHT, INPUT_WIDTH, C_DIM],
                        name='real_images')

    # noise / seed
    z = tf.placeholder(tf.float32, [None, Z_DIM], name='z')
    z_sum = tf.summary.histogram('z', z)

    gen_output = generator(z, y, batch_size=BATCH_SIZE, z_dim=Z_DIM, output_dim=[INPUT_HEIGHT, INPUT_WIDTH],
                            gf_dim=64, gfc_dim=1024, c_dim=C_DIM)

    # actual images as input
    d, d_logits = discriminator(inputs, reuse=False)
    # generated, "fake" images as input
    d_, d_logits_ = discriminator(gen_output, reuse=True)

    d_sum = tf.summary.histogram("d", d)
    d__sum = tf.summary.histogram("d_", d_)
    g_sum = tf.summary.image("G", gen_output)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits, labels=tf.random_uniform(
            d.shape,
            minval=0.7,
            maxval=1.2,
            dtype=tf.float32,
            seed=None,
            name=None
            )))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_, labels=tf.random_uniform(
            d.shape,
            minval=0,
            maxval=0.3,
            dtype=tf.float32,
            seed=None,
            name=None
            )
    ))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_, labels=tf.random_uniform(
            d_.shape,
            minval=0.7,
            maxval=1.2,
            dtype=tf.float32,
            seed=None,
            name=None
            )))

    d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
    d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)

    d_loss = d_loss_fake + d_loss_real

    g_loss_sum = tf.summary.scalar("g_loss_sum", g_loss)
    d_loss_sum = tf.summary.scalar("d_loss_sum", d_loss)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    """ TRAIN PART """

    d_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_1) \
                .minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_1) \
                .minimize(g_loss, var_list=g_vars)

    init_vars()
    saver = tf.train.Saver()

    g_sum = tf.summary.merge([z_sum, d__sum, g_sum, d_loss_fake_sum, g_loss_sum])
    d_sum = tf.summary.merge([z_sum, d_sum, d_loss_real_sum, d_loss_sum])
    writer = tf.summary.FileWriter('./logs', sess.graph)

    if not restore:
        for epoch in xrange(NUM_EPOCHES):
            images, labels = mnist.train.next_batch(BATCH_SIZE)
            images = np.reshape(images, (-1, 28, 28, 1))

            batch_z = np.random.uniform(-1, 1, size=(BATCH_SIZE , Z_DIM))
            _, summary_str = sess.run([d_optim, d_sum], feed_dict={inputs: images, y: labels, z: batch_z})
            if epoch % 100 == 0:
                writer.add_summary(summary_str, epoch)

            _, summary_str = sess.run([g_optim, g_sum], feed_dict={y: labels, z: batch_z})
            if epoch % 100 == 0:
                writer.add_summary(summary_str, epoch)
            _, summary_str = sess.run([g_optim, g_sum], feed_dict={y: labels, z: batch_z})
            if epoch % 100 == 0:
                writer.add_summary(summary_str, epoch)

            if epoch % 100 == 0:
                errD_fake = d_loss_fake.eval({
                    z: batch_z,
                    y: labels
                })
                errD_real = d_loss_real.eval({
                    inputs: images,
                    y: labels
                })
                errG = g_loss.eval({
                    z: batch_z,
                    y: labels
                })
                print("Epoch: [%2d], d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, errD_fake+errD_real, errG))
        save_path = saver.save(sess, "./save/")
    else:
        saver.restore(sess, "./save/")
        print("Model loaded")

        images, labels = mnist.train.next_batch(BATCH_SIZE)
        batch_z = np.random.uniform(-1, 1, size=(BATCH_SIZE , Z_DIM))

        generated_images = gen_output.eval(feed_dict={z: batch_z, y: labels})

        for i, image in enumerate(generated_images):
            image = np.reshape(image, (28, 28))
            plt.imshow(image, vmin=0, vmax=1, cmap='gray')
            plt.savefig('./pictures/' + str(i) + '.png')



with tf.Session() as sess:
    main(sess, restore=False)
