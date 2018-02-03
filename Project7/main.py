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
    NUM_EPOCHES = 100
    INPUT_HEIGHT = 28
    INPUT_WIDTH = 28
    # Color dimension: e.g 1 for grayscale and 3 for RGB
    C_DIM = 1
    # number of dimension of a label
    Y_DIM = 10
    # number of elements in generator conv2d_transpose
    Z_DIM = 100

    # optimizer variables
    LEARNING_RATE = 0.0002
    BETA_1 = 0.5

    # If you want to evaluate: set train=False and Restore=True
    TRAIN = False
    RESTORE = True

    DATA_PATH = './tmp/tensorflow/mnist/mnist_fashion'

    mnist = loadDataFromMNIST(DATA_PATH)

    inputs = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGHT, INPUT_WIDTH, C_DIM],
                        name='real_images')
    y = tf.placeholder(tf.float32, [BATCH_SIZE, Y_DIM], name='y')

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

    labels_normal_zero = tf.zeros_like(d)
    labels_normal_one = tf.ones_like(d_)
    # labels_normal_zero = tf.abs(tf.random_normal(d.shape, stddev=0.3))
    # labels_normal_one = tf.random_normal(d_.shape, mean=1.0, stddev=0.3)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits, labels=labels_normal_one))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_, labels=labels_normal_zero))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_, labels=labels_normal_one))

    # d_loss_real = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits, labels=tf.random_uniform(
    #         d.shape,
    #         minval=0.7,
    #         maxval=1.2,
    #         dtype=tf.float32,
    #         seed=None,
    #         name=None
    #         )))
    # d_loss_fake = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_, labels=tf.random_uniform(
    #         d.shape,
    #         minval=0,
    #         maxval=0.3,
    #         dtype=tf.float32,
    #         seed=None,
    #         name=None
    #         )
    # ))
    # g_loss = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_, labels=tf.random_uniform(
    #         d_.shape,
    #         minval=0.7,
    #         maxval=1.2,
    #         dtype=tf.float32,
    #         seed=None,
    #         name=None
    #         )))

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

    #g_sum = tf.summary.merge([z_sum, d__sum, g_sum, d_loss_fake_sum, g_loss_sum])
    g_sum = tf.summary.merge([g_sum, d_loss_fake_sum, g_loss_sum])

    #d_sum = tf.summary.merge([z_sum, d_sum, d_loss_real_sum, d_loss_sum])
    d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    writer = tf.summary.FileWriter('./logs', sess.graph)

    epoch_of_checkpoint = 0
    if RESTORE:
        #epoch_of_checkpoint = tryToRestoreSavedSession(saver, sess)
        if TRAIN and epoch_of_checkpoint > NUM_EPOCHES:
            print('### WARNING: Max. number of epoches already reached.')
            return
    if TRAIN:
        batches_number = int(mnist.train.num_examples / BATCH_SIZE)
        for epoch in xrange(epoch_of_checkpoint, NUM_EPOCHES):
            iteration = epoch * batches_number
            for batch_number in xrange(batches_number):
                images, labels = mnist.train.next_batch(BATCH_SIZE)
                images = np.reshape(images, (-1, 28, 28, 1))
                batch_z = np.random.uniform(-1, 1, size=(BATCH_SIZE , Z_DIM))

                _, summary_str = sess.run([d_optim, d_sum],
                                feed_dict={inputs: images, y: labels, z: batch_z})
                if batch_number % 100 == 0:
                    writer.add_summary(summary_str, iteration + batch_number)

                _, summary_str = sess.run([g_optim, g_sum],
                                feed_dict={y: labels, z: batch_z})
                if batch_number % 100 == 0:
                    writer.add_summary(summary_str, iteration + batch_number)

                _, summary_str = sess.run([g_optim, g_sum],
                                feed_dict={y: labels, z: batch_z})
                if batch_number % 100 == 0:
                    writer.add_summary(summary_str, iteration + batch_number)

                if batch_number % 100 == 0:
                    errD_fake = d_loss_fake.eval({z: batch_z, y: labels})
                    errD_real = d_loss_real.eval({inputs: images, y: labels})
                    errG = g_loss.eval({z: batch_z, y: labels})
                    print("Epoch: [%2d] [%4d / %4d], d_loss: %.8f, g_loss: %.8f" \
                        % (epoch, batch_number, batches_number, errD_fake+errD_real, errG))

            saver.save(sess, "./save/")
            saveEpochToFile(epoch)

        saver.save(sess, "./save/")
        saveEpochToFile(epoch)

    """ EVALUATING """
    if RESTORE and not TRAIN:
        print('### EVALUATING')
        epoch_of_checkpoint = tryToRestoreSavedSession(saver, sess)
        images, labels = mnist.test.next_batch(BATCH_SIZE)
        batch_z = np.random.uniform(-1, 1, size=(BATCH_SIZE , Z_DIM))

        generated_images = gen_output.eval(feed_dict={z: batch_z, y: labels})
        #generated_images_flat = np.reshape(generated_images, (BATCH_SIZE, -1))
        #saveImageAndNoise(np.concatenate((generated_images_flat, batch_z), axis=1),
        #                                    z_dim=Z_DIM)

        for i in range(len(generated_images)):
            print('### printing image {} of {}'.format(i, len(generated_images)))
            plotImage(generated_images[i], 28, 28, str(i) + '-image')
            plotImage(batch_z[i], 4, 25, str(i) + '-noise')

        # for i, image in enumerate(generated_images):
        #     image = np.reshape(image, (28, 28))
        #     plt.imshow(image, vmin=0, vmax=1, cmap='gray')
        #     plt.savefig('./pictures/' + str(i) + '.png')


with tf.Session() as sess:
    main(sess)

    #images, noises = loadImageAndNoise()
    # for i in range(len(images)):
    #     saveImage(images[i], 28, 28, str(i) + '-image')
    #     saveImage(noises[i], 4, 25, str(i) + '-noise')
