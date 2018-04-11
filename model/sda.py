import tensorflow as tf


class StackedDenosingAutoencoder(object):
    """Stacked Denoising Autoencoder

    This class implements the StackedDenoisingAutoencoder (SDA) model using tensorflow operations.
    It supports three nosing approaches: Additive White Gaussian, Masking, and Salt-and-Pepper.

    Author: Lucas Venezian Povoa (lucasvenez@gmail.com)

    Example:
        sda = StackedDenoisingAutoencoder()

        sda.build(n_neurons_per_encoder=(1024, 512, 256), noise_approach

    References:
        [1] Stacked Denoising Autoencoders: Learning Useful Representations in a Deep
        Network with a Local Denoising Criterion

        [2] https://en.wikipedia.org/wiki/Additive_white_Gaussian_noise
    """
    def __init__(self, session=None):

        self.graph = tf.Graph() if session is None else session.graph

        self.sess = session if session is not None else tf.Session(graph=self.graph)

        self.inputs, self.encoders, self.decoders, self.losses = [], [], [], []

        #
        # Placeholders
        #
        self.input_ph = None

    def build(self, n_inputs, n_neurons_per_encoder=(1024, 512, 256)):

        assert isinstance(n_neurons_per_encoder, tuple)

        with self.graph.as_default():

            with tf.name_scope('stacked_denoising_autoencoder'):

                self.inputs += [tf.placeholder(dtype=tf.float32, shape=(None, n_inputs), name='input_encoder_1')]

                n_outputs = n_inputs

                for index, n_neurons in enumerate(n_neurons_per_encoder):

                    self.encoders += self.dense(self.inputs[-1], n_neurons, activation_function=tf.nn.tanh,
                                                name='encoder_{}'.format(index + 1))

                    self.decoders += self.dense(self.encoders[-1], n_outputs,
                                                name='decoder_{}'.format(index + 1))

                    if index < len(n_neurons_per_encoder) - 1:

                        self.inputs += [tf.placeholder(dtype=tf.float32, shape=(None, n_neurons),
                                                       name='input_encoder_{}'.format(index + 2))]

                        n_outputs = n_neurons

        return self.encoders, self.decoders

    def build_optimizers(self):

        with tf.name_scope('optimizers'):
            pass

    def predict(self, x):

        with self.graph.as_default():
            pass

    def optimize(self, x, noise_weight=.5, noise_approach='awgn', noise_fraction=.2):

        with self.graph.as_default():
            pass

    def dense(self, input, n_neurons, activation_function, name):

        assert name is not None

        with self.graph.as_default():

            w = tf.Variable(tf.truncated_normal([input.shape[1], n_neurons], stddev=.1), name='{}_weights'.format(name))

            b = tf.Variable(tf.truncated_normal([n_neurons], stddev=.1), name='{}_biases'.format(name))

            if activation_function is not None:
                return activation_function(tf.add(tf.matmul(input, w), b), name=name)

            else:
                return tf.add(tf.matmul(input, w), b, name=name)
