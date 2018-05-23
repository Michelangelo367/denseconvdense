import os
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",category=DeprecationWarning)

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import util
import os
import numpy as np


class Dense(object):

    VALID_ACTIVATION_FUNCTIONS = {'sigmoid': tf.nn.sigmoid, 'tanh': tf.nn.tanh, 'relu': tf.nn.relu}

    VALID_OPTIMIZERS = {'sgd': tf.train.GradientDescentOptimizer, 'ftrl': tf.train.FtrlOptimizer,
                        'adam': tf.train.AdamOptimizer, 'adadelta': tf.train.AdadeltaOptimizer,
                        'adagrad': tf.train.AdagradOptimizer, 'rmsprop': tf.train.RMSPropOptimizer}

    VALID_COST_FUNCTIONS = ('softmax_cross_entroy')

    def __init__(self, session=None, summaries_dir='../log', model_name='M0000'):

        self.graph = tf.Graph()

        if session is None:
            self.sess = tf.Session(graph=self.graph)

        self.summaries_dir = summaries_dir

        self.saver = None

        self.model_name = model_name

        self.phase = None

    def init(self,  n_input_features, n_outputs, abstraction_activation_functions=('sigmoid', 'tanh', 'relu'),
             n_hidden_layers=3, n_hidden_nodes=10, keep_probability=0.5, initialization='RBM',
             optimizer_algorithms=('sgd', 'sgd', 'sgd'), cost_function='softmax_cross_entropy', add_summaries=True,
             batch_normalization=False):

        assert isinstance(n_hidden_nodes, int) and isinstance(abstraction_activation_functions, tuple)

        assert 0. < keep_probability <= 1.

        assert n_hidden_nodes > 0 and n_hidden_layers > 0

        assert len(optimizer_algorithms) == len(abstraction_activation_functions)

        with self.graph.as_default():

            self.n_input_features = n_input_features

            self.abstraction_activation_functions = abstraction_activation_functions

            self.n_hidden_nodes = n_hidden_nodes

            self.n_hidden_layers = n_hidden_layers

            self.keep_probability = keep_probability

            self.phase = tf.placeholder(tf.bool, name='phase_ph')

            self.n_outputs = n_outputs

            self.optimizer_algorithms = optimizer_algorithms

            self.cost_function = cost_function

            self.lr = None

            self.add_summaries = add_summaries

            self.batch_normalization = batch_normalization

            #
            # TODO It is not used, yet!
            #
            self.initialization = initialization

            #
            # Placeholders
            #
            self.raw_input = None

            self.expected_output = None

            self.model_path = None

            self.keep_prob = None

            #
            #
            #
            self.models = [None for _ in range(len(abstraction_activation_functions))]

            self.cost_functions = [None for _ in range(len(abstraction_activation_functions))]

            self.optimizers = [None for _ in range(len(abstraction_activation_functions))]

            self.correct_predictions = [None for _ in range(len(abstraction_activation_functions))]

            self.accuracies = [None for _ in range(len(abstraction_activation_functions))]

            self.abstract_representation = [[None for _ in range(n_hidden_layers)]
                                             for _ in range(len(abstraction_activation_functions))]

            self.test_writer = None

            self.merged = None

    def optimize(self, x, y, x_test=None, y_test=None, learning_rate=1e-5, steps=1000, batch_size=1000,
                 weights=1.0, shuffle=True):

        assert steps > 0

        assert 0 < batch_size <= x.shape[0]

        self.build_optimizers(weights=weights)

        print('Optimizing model')

        if batch_size is None:
            batch_size = x.shape[0]

        if x_test is None:
            x_test = x

        if y_test is None:
            y_test = y

        #
        # FIXME check if there is not unitialized variables
        #
        with self.graph.as_default():

            self.sess.run(tf.global_variables_initializer())

            self.sess.run(tf.local_variables_initializer())

            test_writer = tf.summary.FileWriter(self.summaries_dir + '/{}'.format(self.model_name), tf.get_default_graph())

            n_rows = x.shape[0]

            index = np.array(list(range(n_rows)), dtype=np.int)

            j = 0

            for step in range(steps):

                print('Optimization at step {}'.format(step + 1))

                current_block = 0

                while (current_block < n_rows):

                    if shuffle:
                        np.random.shuffle(index)

                    batch = list(range(current_block, (min(current_block + batch_size, n_rows))))

                    self.sess.run([self.merged] + self.optimizers,
                                  feed_dict={self.raw_input: x[index[batch], :],
                                             self.expected_output: y[index[batch], :],
                                             self.keep_prob: self.keep_probability,
                                             self.lr: learning_rate, self.phase: True})

                    current_block += batch_size

                    j += 1

                test_results = self.sess.run([self.merged],
                                             feed_dict={self.raw_input: x_test,
                                                        self.expected_output: y_test,
                                                        self.keep_prob: 1., self.phase: False})

                self.saver.save(self.sess, '../output/{0}/{0}'.format(self.model_name), global_step=step)

                if self.add_summaries:
                    test_writer.add_summary(test_results[0], step)

    def predict(self, x):

        with self.graph.as_default():

            result = self.sess.run(self.abstract_representation, feed_dict={self.raw_input: x, self.keep_prob: 1., self.phase: False})

            result = np.array(result)

            result = np.rollaxis(np.rollaxis(result, 2, 0), 3, 2)

            return np.reshape(result, (-1, result.shape[1], result.shape[2], result.shape[3], 1))

    def predict__(self, x):

        with self.graph.as_default():

            result = self.sess.run(self.models, feed_dict={self.raw_input: x, self.keep_prob: 1., self.phase: False})

            result = np.array(result)

            return result

    def load(self, model_path):

        if os.path.exists('{}.meta'.format(model_path)) and os.path.isfile('{}.meta'.format(model_path)):

            with self.graph.as_default():

                self.saver = tf.train.import_meta_graph('{}.meta'.format(model_path))

                self.saver.restore(self.sess, tf.train.latest_checkpoint(os.path.dirname(model_path)))

                self.raw_input = tf.get_default_graph().get_tensor_by_name('raw_input:0')

                self.expected_output = tf.get_default_graph().get_tensor_by_name('expected_output:0')

                self.keep_prob = tf.get_default_graph().get_tensor_by_name('dropout_keep_probability:0')

                self.phase = tf.get_default_graph().get_tensor_by_name('phase_ph:0')

                self.models = [tf.get_default_graph().get_tensor_by_name(n.name +':0') for n in tf.get_default_graph().get_operations()
                               if 'dense_model_' in n.name.split('/')[-1]]

                self.abstract_representation = []

                for model in self.models:

                    model_function = model.name.split('_')[-1].split(':')[0]

                    self.abstract_representation.append([tf.get_default_graph().get_tensor_by_name(op.name + ':0')
                                                         for op in tf.get_default_graph().get_operations()
                                                         if 'hidden_{0}_layer_'.format(model_function) in op.name and
                                                         '/{}'.format(model_function.title()) in op.name and
                                                         'grad' not in op.name])

                self.n_hidden_layers = len(self.abstract_representation[-1])

                self.n_hidden_nodes = self.abstract_representation[-1][-1].shape[1]

        else:
            print('Invalid model path: {}'.format(model_path))

    def build(self, n_features, n_outputs, abstraction_activation_functions,
              n_hidden_layers, n_hidden_nodes, keep_probability, initialization,
              optimizer_algorithms, cost_function='softmax_cross_entropy', add_summaries=True,
              batch_normalization=True):

        self.init(n_features, n_outputs, abstraction_activation_functions,
                  n_hidden_layers, n_hidden_nodes, keep_probability, initialization,
                  optimizer_algorithms, cost_function, add_summaries,
                  batch_normalization)

        with self.graph.as_default():

            self.raw_input = tf.placeholder(tf.float32, shape=(None, self.n_input_features), name='raw_input')

            self.expected_output = tf.placeholder(tf.float32, shape=(None, self.n_outputs), name='expected_output')

            self.keep_prob = tf.placeholder(tf.float32, name='dropout_keep_probability')

            with tf.name_scope('abstraction_layer'):

                for i, activation_function in enumerate(self.abstraction_activation_functions):

                    with tf.name_scope('{}_model'.format(activation_function[:4])):

                        previous_layer_size, previous_layer = self.n_input_features, self.raw_input

                        for j in range(self.n_hidden_layers):

                            layer_name = 'hidden_{}_layer_{}'.format(activation_function[:4], j + 1)

                            with tf.name_scope(layer_name):
                                #
                                # TODO refactor code to define a function to create dense layers
                                #
                                af = self.VALID_ACTIVATION_FUNCTIONS[activation_function]

                                weight_name = 'weight_{}_h{}{}'.format(activation_function[:4], i + 1, j + 1)

                                w = tf.Variable(tf.truncated_normal([previous_layer_size, self.n_hidden_nodes], stddev=.1), name=weight_name)

                                bias_name = 'bias_{}_h{}{}'.format(activation_function[:4], i + 1, j + 1)

                                b = tf.Variable(tf.truncated_normal([self.n_hidden_nodes], stddev=.1), name=bias_name)

                                abstraction_layer_name = 'abstraction_{}_layer_{}'.format(activation_function[:4], j + 1)

                                self.abstract_representation[i][j] = \
                                    tf.nn.dropout(af(tf.add(tf.matmul(previous_layer, w), b)), self.keep_prob,
                                                  name=abstraction_layer_name if not self.batch_normalization
                                                        else 'dropout_{}_{}'.format(activation_function[:4], j + 1))

                                if self.batch_normalization:
                                    self.abstract_representation[i][j] = \
                                        tf.layers.batch_normalization(self.abstract_representation[i][j], training=self.phase,
                                                                      name=abstraction_layer_name)

                                previous_layer, previous_layer_size = self.abstract_representation[i][j], self.n_hidden_nodes

                                if self.add_summaries:
                                    util.create_tf_scalar_summaries(w, 'weights')
                                    util.create_tf_scalar_summaries(b, 'biases')
                                    util.create_tf_scalar_summaries(self.abstract_representation[i][j], 'activation')

                        with tf.name_scope('output_{}_layer'.format(activation_function[:4])):

                            weight_name = 'weight_{}_out'.format(activation_function[:4])

                            w = tf.Variable(tf.truncated_normal([previous_layer_size, self.n_outputs], stddev=.1), name=weight_name)

                            bias_name = 'bias_{}_out'.format(activation_function[:4])

                            b = tf.Variable(tf.truncated_normal([self.n_outputs], stddev=.1), name=bias_name)

                            dense_name = 'dense_model_{}'.format(activation_function[:4])

                            self.models[i] = tf.add(tf.matmul(previous_layer, w), b, name=dense_name)

                            if self.add_summaries:
                                util.create_tf_scalar_summaries(w, 'weights')
                                util.create_tf_scalar_summaries(b, 'biases')
                                util.create_tf_scalar_summaries(self.models[i], 'output')

            self.saver = tf.train.Saver()

    def build_optimizers(self, weights=1.0):

        print('Building optimizers')

        with self.graph.as_default():

            if self.lr is None:
                self.lr = tf.placeholder(tf.float32, name='learning_rate')

            self.confusion_update = []

            for i, (model, optimizer, activation_function) in \
                    enumerate(zip(self.models, self.optimizer_algorithms, self.abstraction_activation_functions)):

                if self.cost_function == 'softmax_cross_entropy':

                    with tf.name_scope('optimization_{}'.format(activation_function)):

                        self.cost_functions[i] = tf.nn.weighted_cross_entropy_with_logits(
                            targets=self.expected_output, logits=model, pos_weight=np.array(weights))

                        self.optimizers[i] = self.VALID_OPTIMIZERS[optimizer](learning_rate=self.lr).minimize(self.cost_functions[i])

                        if self.add_summaries:
                            tf.summary.scalar('cross_entropy_{}'.format(activation_function), tf.reduce_mean(self.cost_functions[i]))

                else:
                    raise ValueError('Only softmax_cross_entropy cost function is supported, yet.')

                #
                # TODO Add new performance metrics
                #
                with tf.name_scope('evaluation_{}'.format(activation_function)):

                    with tf.name_scope('correct_prediction_{}'.format(activation_function)):
                        self.correct_predictions[i] = tf.equal(tf.argmax(model, 1), tf.argmax(self.expected_output, 1))
                        #self.correct_predictions[i] = tf.logical_or(
                        #    tf.logical_and(tf.greater_equal(model, .5), tf.equal(self.expected_output, 1)),
                        #    tf.logical_and(tf.less(model, .5), tf.equal(self.expected_output, 0)))

                    with tf.name_scope('accuracy_{}'.format(activation_function)):
                        self.accuracies[i] = tf.reduce_mean(tf.cast(self.correct_predictions[i], tf.float32))

                if self.add_summaries:

                    tf.summary.scalar('accuracy_{}'.format(activation_function), self.accuracies[i])

                    #
                    # Create summary tensors
                    #
                    self.merged = tf.summary.merge_all()

