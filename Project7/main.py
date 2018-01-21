from discriminator import *
from generator import *
from ops import loadDataFromMNIST


# size of eaach picture: 28 x 28
def main(sess):
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
    LEARNING_RATE = 0.002
    BETA_1 = 0.5

    mnist = loadDataFromMNIST()
    images, labels = mnist.train.next_batch(BATCH_SIZE)

    #x = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_WIDTH*INPUT_HEIGHT])
    y = tf.placeholder(tf.float32, [BATCH_SIZE, Y_DIM], name='y')

    inputs = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGHT, INPUT_WIDTH, C_DIM],
                        name='real_images')

    z = tf.placeholder(tf.float32, [None, Z_DIM], name='z')
    z_sum = tf.histogram_summary('z', z)

    gen_output = generator(z, y, z_dim=Z_DIM, output_dim=[INPUT_HEIGHT, INPUT_WIDTH],
                            gf_dim=64, gfc_dim=1024, c_dim=C_DIM)

    d, d_logits = discriminator(inputs, y, reuse=False)
    d_, d_logits_ = discriminator(gen_output, y, reuse=True)

    d_sum = histogram_summary("d", d)
    d__sum = histogram_summary("d_", d_)
    g_sum = image_summary("G", gen_output)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits, tf.ones_like(d)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(d_logits_, tf.ones_like(d_)))
    gen_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=, labels=))

    d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
    d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)

    d_loss = d_loss_fake + d_loss_real_sum

    g_loss_sum = tf.summary.scalar("g_loss_sum", gen_loss)
    d_loss_sum = tf.summary.scalar("d_loss_sum", d_loss)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    """ TRAIN PART """

    d_optimizer = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_1) \
                .minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                .minimize(self.g_loss, var_list=self.g_vars)

    init_vars()

    g_sum = tf.merge_summary([z_sum, d__sum, g_sum, d_loss_fake_sum, g_loss_sum])
    d_sum = tf.merge_summary([z_sum, d_sum, d_loss_real_sum, d_loss_sum])
    writer = tf.train.SummaryWriter('./logs', sess.graph)

    for epoch in xrange(NUM_EPOCHES):
        pass




with tf.Session() as sess:
    main(sess)
