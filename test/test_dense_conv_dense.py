import unittest
import pandas as pd

from model import Dense, ConvDense, ConvDense2


class TestDenseConvDense(unittest.TestCase):

    def setUp(self):

        train = pd.read_table('../input/mnist/train.csv', sep=',')

        self.train_x = train.iloc[:, 1:].as_matrix() / 255.
        self.train_y = pd.get_dummies(train.iloc[:, 0]).as_matrix().astype(int)

        self.test_x = pd.read_table('../input/mnist/test.csv', sep=',').as_matrix() / 255.

    def test_model(self):

        self.model = Dense(model_name='M0512-normalized-sigmoid-tanh-relu-adagrad-rmsprop-rmsprop')

        self.model.build(n_input_features=self.train_x.shape[1], n_outputs=self.train_y.shape[1],
                         abstraction_activation_functions=('sigmoid', 'tanh', 'relu'),
                         n_hidden_layers=3, optimizer_algorithms=('adagrad', 'rmsprop', 'rmsprop'),
                         n_hidden_nodes=512, keep_probability=0.5,
                         initialization='RBM', batch_normalization=True)

        self.model.optimize(self.train_x, self.train_y, learning_rate=1e-3)

    def test_load(self):

        self.model = Dense()

        self.model.load('../output/model/M0000/M0000-999')

        transformed_x = self.model.predict(self.train_x)

        self.assertEqual(transformed_x.shape, (40000, 3, 100, 3))

    def test_conv_dense(self):

        self.model = Dense()
        self.model.load('../output/{0}/{0}-999'.format('M0512-normalized-sigmoid-tanh-relu-adagrad-rmsprop-rmsprop'))

        self.train_x = self.model.predict(self.train_x).reshape((-1, 3, 512, 3, 1))
        self.test_x = self.model.predict(self.test_x).reshape((-1, 3, 512, 3, 1))

        del self.model

        model_name = 'C0512-nrm-maxp-relu-512f-adadelta-huber-50steps-500bs'

        self.conv = ConvDense(model_name=model_name)
        self.conv.build(n_models=3, n_neurons_per_layer=512, optimizer_algorithm='adadelta')

        self.conv.optimize(x=self.train_x, y=self.train_y, x_test=self.train_x[41000:,:], y_test=self.train_y[41000:,:],
                           learning_rate=.25, steps=50, batch_size=500)

        y = self.conv.predict(self.test_x)

        pd.DataFrame({'ImageId': list(range(1, len(y) + 1)), 'Label': y}).to_csv(
            '../output/{}.csv'.format(model_name), sep=',', index=False)

    def test_conv_dense2(self):

        self.model = Dense()
        self.model.load('../output/M0512/M0512-999')

        self.train_x = self.model.predict(self.train_x).reshape((-1, 3, 512, 3, 1))
        self.test_x = self.model.predict(self.test_x).reshape((-1, 3, 512, 3, 1))

        del self.model

        model_name = 'C0512-2-adagrad-huber-50steps-2000bs'

        self.conv = ConvDense2(model_name=model_name)
        self.conv.build(n_models=3, n_neurons_per_layer=512)

        self.conv.optimize(x=self.train_x, y=self.train_y, x_test=self.train_x[41000:,:], y_test=self.train_y[41000:,:],
                           learning_rate=.25, steps=50, batch_size=500)

        y = self.conv.predict(self.test_x)

        pd.DataFrame({'ImageId': list(range(1, len(y) + 1)), 'Label': y}).to_csv(
            '../output/{}.csv'.format(model_name), sep=',', index=False)
