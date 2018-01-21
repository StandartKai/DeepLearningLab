from discriminator import *
from generator import *
from ops import loadDataFromMNIST


# size of eaach picture: 28 x 28
def main():
    BATCH_SIZE = 64
    INPUT_HEIGHT = 28
    INPUT_WIDTH = 28
    # Color dimension: e.g 1 for grayscale and 3 for RGB
    C_DIM = 1

    mnist = loadDataFromMNIST()
    images, labels = mnist.train.next_batch(BATCH_SIZE)

    x = tf.placeholder(tf.float32, [None, INPUT_WIDTH*INPUT_HEIGHT])
    # inputs = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGHT, INPUT_WIDTH, C_DIM])
    z = tf.placeholder(tf.float32, [None, C_DIM])

    gen_output = generator(z, z_dim=100, output_dim=[INPUT_HEIGHT, INPUT_WIDTH],
                            gf_dim=64, gfc_dim=1024, c_dim=C_DIM)


main()
