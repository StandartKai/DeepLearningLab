from discriminator import *
from generator import *
from ops import loadDataFromMNIST
from ops import groupLabels

import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from six.moves import xrange

# size of eaach picture: 28 x 28
def main(sess, batch_size, num_epochs, input_height, input_width, c_dim, y_dim,
         z_dim, learning_rate, beta_1, data_path, train, restore):

    print("________________________________________")
    print("called main with settings: ")
    print("________________________________________")
    print("batch_size: " + str(batch_size))
    print("num_epochs: " + str(num_epochs))
    print("input_height: " + str(input_height))
    print("input_width: " + str(input_width))
    print("c_dim: " + str(c_dim))
    print("y_dim: " + str(y_dim))
    print("z_dim: " + str(z_dim))
    print("learning_rate: " + str(learning_rate))
    print("beta_1: " + str(beta_1))
    print("data_path: " + str(data_path))
    print("train: " + str(train))
    print("restore: " + str(restore))
    print("________________________________________")


    mnist = loadDataFromMNIST()

    inputs = tf.placeholder(tf.float32, [batch_size, input_height, input_width, c_dim],
                        name='real_images')
    y = tf.placeholder(tf.float32, [batch_size, y_dim], name='y')

    # noise / seed
    z = tf.placeholder(tf.float32, [None, z_dim], name='z')
    z_sum = tf.summary.histogram('z', z)

    gen_output = generator(z, y, batch_size=batch_size, z_dim=z_dim, output_dim=[input_height, input_width],
                            gf_dim=64, gfc_dim=1024, c_dim=c_dim)

    # actual images as input
    d, d_logits = discriminator(inputs, reuse=False)
    # generated, "fake" images as input
    d_, d_logits_ = discriminator(gen_output, reuse=True)

    d_sum = tf.summary.histogram("d", d)
    d__sum = tf.summary.histogram("d_", d_)
    g_sum = tf.summary.image("G", gen_output)

    labels_normal_zero = tf.zeros_like(d)
    labels_normal_one = tf.ones_like(d_)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits, labels=labels_normal_one))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_, labels=labels_normal_zero))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_, labels=labels_normal_one))


    d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
    d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)

    d_loss = d_loss_fake + d_loss_real

    g_loss_sum = tf.summary.scalar("g_loss_sum", g_loss)
    d_loss_sum = tf.summary.scalar("d_loss_sum", d_loss)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    """ TRAIN PART """

    d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta_1) \
                .minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta_1) \
                .minimize(g_loss, var_list=g_vars)

    init_vars()
    saver = tf.train.Saver()

    #g_sum = tf.summary.merge([z_sum, d__sum, g_sum, d_loss_fake_sum, g_loss_sum])
    g_sum = tf.summary.merge([g_sum, d_loss_fake_sum, g_loss_sum])

    #d_sum = tf.summary.merge([z_sum, d_sum, d_loss_real_sum, d_loss_sum])
    d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    writer = tf.summary.FileWriter('./logs', sess.graph)

    epoch_of_checkpoint = 0
    if restore:
        #epoch_of_checkpoint = tryTorestoreSavedSession(saver, sess)
        if train and epoch_of_checkpoint > NUM_EPOCHES:
            print('### WARNING: Max. number of epoches already reached.')
            return
    if train:
        # batches_number = int(mnist.train.num_examples / batch_size)
        images_train, labels_train = extractShirts(mnist)
        batches_number = int(images_train.shape[0] / batch_size)

        for epoch in xrange(epoch_of_checkpoint, NUM_EPOCHES):
            iteration = epoch * batches_number
            for batch_number in xrange(batches_number):
                # images, labels = mnist.train.next_batch(batch_size)
                # hacker detected:
                if batch_number != (batches_number - 1):
                    images_batch = images_train[batch_number*batch_size:(batch_number+1)*batch_size]
                    labels_batch = labels_train[batch_number*batch_size:(batch_number+1)*batch_size]
                else:
                    images_batch = images_train[batch_number*batch_size:]
                    labels_batch = labels_train[batch_number*batch_size:]

                images = np.reshape(images_batch, (-1, input_width, input_height, 1))
                labels = labels_batch

                batch_z = np.random.uniform(-1, 1, size=(batch_size , z_dim))

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
    if restore and not train:
        print('### EVALUATING')
        epoch_of_checkpoint = tryToRestoreSavedSession(saver, sess)
        images, labels = mnist.test.next_batch(batch_size)
        batch_z = np.random.uniform(-1, 1, size=(batch_size , z_dim))

        generated_images = gen_output.eval(feed_dict={z: batch_z, y: labels})

        for i in range(len(generated_images)):
            print('### printing image {} of {}'.format(i, len(generated_images)))
            plotImage(generated_images[i], 28, 28, str(i) + '-image')
            plotImage(batch_z[i], 4, 25, str(i) + '-noise')


with tf.Session() as sess:
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', metavar='BS', type=int, nargs='?', help='batch size')
    parser.add_argument('num_epochs', metavar='NE', type=int, nargs='?', help='number of epochs')
    parser.add_argument('input_height', metavar='IH', type=int, nargs='?', help='input height')
    parser.add_argument('input_width', metavar='IW', type=int, nargs='?', help='input width')

    parser.add_argument('c_dim', metavar='CDIM', type=int, nargs='?', help='color dimension')
    parser.add_argument('y_dim', metavar='YDIM', type=int, nargs='?', help='label dimension')
    parser.add_argument('z_dim', metavar='ZDIM', type=int, nargs='?', help='noise dimension')

    parser.add_argument('learning_rate', metavar='LR', type=int, nargs='?', help='learning rate')
    parser.add_argument('beta_1', metavar='B1', type=int, nargs='?', help='beta_1 learning rate')

    parser.add_argument('data_path', metavar='PATH', type=str, nargs='?', help='path to the dataset')

    parser.add_argument('train', metavar='train', type=bool, nargs='?', help='true if you want to train')
    parser.add_argument('restore', metavar='restore', type=bool, nargs='?', help='true if you want to restore')

    args = parser.parse_args()

    # set parameters on custom or default values if not given

    batch_size = 64 if not args.batch_size else args.batch_size
    num_epochs = 100 if not args.num_epochs else args.num_epochs
    input_height = 28 if not args.input_height else args.input_height
    input_width = 28 if not args.input_width else args.input_width

    # Color dimension: e.g 1 for grayscale and 3 for RGB
    c_dim = 1 if not args.c_dim else args.c_dim
    # number of dimension of a label
    y_dim = 10 if not args.y_dim else args.y_dim
    # number of elements in generator conv2d_transpose
    z_dim = 100 if not args.z_dim else args.z_dim

    # optimizer variables
    learning_rate = 0.0002 if not args.learning_rate else args.learning_rate
    beta_1 = 0.5 if not args.beta_1 else args.beta_1

    data_path = './tmp/tensorflow/mnist/mnist_fashion' if not args.data_path else args.data_path

    train = False if not args.train else args.train
    restore = True if not args.restore else args.restore

    main(sess, batch_size, num_epochs, input_height, input_width, c_dim, y_dim,
         z_dim, learning_rate, beta_1, data_path, train, restore)
