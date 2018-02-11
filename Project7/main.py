import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import distutils.util as utils

from discriminator import *
from generator import *
from ops import loadDataFromMNIST
from ops import groupLabels


from six.moves import xrange


# size of eaach picture: 28 x 28
def main(sess, batch_size, num_epochs, input_height, input_width, c_dim, y_dim,
         z_dim, learning_rate, beta_1, data_type, data_path, train, restore,
         save_dir):

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
    print("data_type: " + str(data_type))
    print("data_path: " + str(data_path))
    print("train: " + str(train))
    print("restore: " + str(restore))
    print("save_dir: " + save_dir)
    print("________________________________________")

    if save_dir != "" and (save_dir[0] == "/" or save_dir[0] == "\\"):
        save_dir = save_dir[1:]

    if (data_type == 'mnist' or data_type == 'mnist_fashion'):
        mnist = loadDataFromMNIST(data_path)


    inputs = tf.placeholder(tf.float32, [batch_size, input_height, input_width, c_dim],
                        name='real_images')
    y = tf.placeholder(tf.float32, [batch_size, y_dim], name='y')

    # noise / seed
    z = tf.placeholder(tf.float32, [None, z_dim], name='z')
    z_sum = tf.summary.histogram('z', z)

    gen_output, gen_intermediate = generator(z, batch_size=batch_size, z_dim=z_dim, output_dim=[input_height, input_width],
                            gf_dim=64, gfc_dim=1024, c_dim=c_dim)

    # actual images as input
    d, d_logits, d_intermediate = discriminator(inputs, reuse=False,
                                            use_minibatch=True)
    # generated, "fake" images as input
    d_, d_logits_, d_intermediate_ = discriminator(gen_output, reuse=True,
                                            use_minibatch=True)

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

    g_cost = tf.sqrt(tf.reduce_sum(tf.pow(d_intermediate - d_intermediate_, 2)))
    g_cost_sum = tf.summary.scalar("g_cost_sum", g_cost)

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
    # g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta_1) \
                # .minimize(g_loss, var_list=g_vars)
    g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta_1) \
                .minimize(g_cost, var_list=g_vars)

    init_vars()
    saver = tf.train.Saver()

    #g_sum = tf.summary.merge([z_sum, d__sum, g_sum, d_loss_fake_sum, g_loss_sum])
    g_sum = tf.summary.merge([g_sum, g_cost_sum, d_loss_fake_sum, g_loss_sum])

    #d_sum = tf.summary.merge([z_sum, d_sum, d_loss_real_sum, d_loss_sum])
    d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    writer = tf.summary.FileWriter('./logs' + save_dir, sess.graph)

    epoch_of_checkpoint = 0
    if restore:
        epoch_of_checkpoint = tryToRestoreSavedSession(saver, sess, save_dir)
        if train and epoch_of_checkpoint > num_epochs:
            print('### WARNING: Max. number of epochs already reached.')
            return
    if train:
        images_train, labels_train = None, None
        batches_number = 0
        if (data_type == 'mnist'):
            batches_number = int(mnist.train.num_examples / batch_size)
        if (data_type == 'mnist_fashion'):
            images_train, labels_train = extractShirts(mnist)
            batches_number = int(images_train.shape[0] / batch_size)
            print("Shirts extracted")
        if (data_type == 'deep_fashion'):
            #images_train, _ = loadData("data_fashion.h5", "images_fashion", False)
            images_train, _ = loadData(data_path, "images_fashion", False)
            labels_train = np.ones((images_train.shape[0], 1))
            batches_number = int(images_train.shape[0] / batch_size)

        if (data_type != 'mnist'):
            assert images_train is not None
            assert labels_train is not None
        assert batches_number != 0

        for epoch in xrange(epoch_of_checkpoint, num_epochs):
            iteration = epoch * batches_number
            for batch_number in xrange(batches_number):
                if (data_type == 'mnist'):
                    images_batch, labels_batch = mnist.train.next_batch(batch_size)
                else:
                    images_batch = images_train[batch_number*batch_size:(batch_number+1)*batch_size]
                    labels_batch = labels_train[batch_number*batch_size:(batch_number+1)*batch_size]


                images_batch = np.reshape(images_batch, (-1, input_width, input_height, c_dim))

                batch_z = np.random.uniform(-1, 1, size=(batch_size , z_dim))

                _, summary_str = sess.run([d_optim, d_sum],
                                feed_dict={inputs: images_batch, y: labels_batch, z: batch_z})
                if batch_number % 100 == 0:
                    writer.add_summary(summary_str, iteration + batch_number)

                _, summary_str = sess.run([g_optim, g_sum],
                                feed_dict={z: batch_z, inputs: images_batch})

                if batch_number % 100 == 0:
                    writer.add_summary(summary_str, iteration + batch_number)

                _, summary_str = sess.run([g_optim, g_sum],
                                feed_dict={z: batch_z, inputs: images_batch})
                if batch_number % 100 == 0:
                    writer.add_summary(summary_str, iteration + batch_number)

                if batch_number % 100 == 0:
                    errD_fake = d_loss_fake.eval({z: batch_z, y: labels_batch})
                    errD_real = d_loss_real.eval({inputs: images_batch, y: labels_batch})
                    errG = g_loss.eval({z: batch_z, y: labels_batch})
                    errG_cost = g_cost.eval({inputs: images_batch, z: batch_z, y: labels_batch})
                    print("Epoch: [%2d] [%4d / %4d], d_loss: %.8f, g_loss: %.8f, g_cost: %.8f" \
                        % (epoch, batch_number, batches_number, errD_fake+errD_real, errG, errG_cost))

            saver.save(sess, "./save/" + save_dir)
            saveEpochToFile(epoch, save_dir)

        saver.save(sess, "./save/" + save_dir)
        saveEpochToFile(epoch, save_dir)

    """ EVALUATING """
    if restore and not train:
        print('### EVALUATING')
        assert z_dim % 4 == 0

        batch_z = np.random.uniform(-1, 1, size=(batch_size , z_dim))
        generated_images = gen_output.eval(feed_dict={z: batch_z})

        batch_z_flat = np.reshape(batch_z, (-1, 4, int(z_dim/4)))
        if c_dim == 1:
            generated_images_flat = np.reshape(
                    generated_images, (-1, input_height, input_height))
        else:
            generated_images_flat = np.reshape(
                generated_images, (-1, input_height, input_height, c_dim))

        for i in range(len(generated_images)):
            print('### printing image {} of {}'.format(i, len(generated_images_flat)))
            plotImage(generated_images_flat[i], str(i) + '-image')
            plotImage(batch_z_flat[i], str(i) + '-noise')


with tf.Session() as sess:
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", default=64)
    parser.add_argument("-ne", "--num_epochs", default=100)
    parser.add_argument("-ih", "--input_height", default=28)
    parser.add_argument("-iw", "--input_width", default=28)
    parser.add_argument("-cd", "--c_dim", default=1)
    parser.add_argument("-yd", "--y_dim", default=10)
    parser.add_argument("-zd", "--z_dim", default=100)
    parser.add_argument("-lr", "--learning_rate", default=0.0002)
    parser.add_argument("-b", "--beta_1", default=0.5)
    parser.add_argument("-dt", "--data_type", type=str, default='mnist_fashion')
    parser.add_argument("-dp", "--data_path", default='./tmp/tensorflow/mnist/mnist_fashion')
    parser.add_argument("-tr", "--train", type=utils.strtobool, default=False)
    parser.add_argument("-re", "--restore", type=utils.strtobool, default=True)
    parser.add_argument("-sd", "--save_dir", type=str, default="")

    args = parser.parse_args()

    main(sess, int(args.batch_size), int(args.num_epochs), int(args.input_height),
        int(args.input_width), int(args.c_dim), int(args.y_dim),
        int(args.z_dim), float(args.learning_rate), float(args.beta_1), args.data_type,
        args.data_path, args.train, args.restore, args.save_dir)
