import unittest
import pandas as pd
import numpy as np

from model import Dense, ConvDense


class TestDenseConvDense(unittest.TestCase):

    def setUp(self):

        train = pd.read_table('../input/mnist/train.csv', sep=',')

        self.train_x = train.iloc[:, 1:].as_matrix()
        self.train_y = pd.get_dummies(train.iloc[:, 0]).as_matrix().astype(int)

        self.test_x = pd.read_table('../input/mnist/test.csv', sep=',').as_matrix()


    def test_model(self):

        self.model = Dense(model_name='M1024')

        self.model.build(n_input_features=self.train_x.shape[1], n_outputs=self.train_y.shape[1],
                         abstraction_activation_functions=('sigmoid', 'tanh', 'relu'),
                         n_hidden_layers=3, optimizer_algorithms=('sgd', 'sgd', 'sgd'),
                         n_hidden_nodes=1024, keep_probability=0.5,
                         initialization='RBM', batch_normalization=True)

        self.model.optimize(self.train_x, self.train_y, learning_rate=1e-3)

    def test_load(self):

        self.model = Dense()

        self.model.load('../output/model/M0000/M0000-999')

        transformed_x = self.model.predict(self.train_x)

        self.assertEqual(transformed_x.shape, (40000, 3, 100, 3))

    def test_conv_dense(self):

        self.model = Dense()
        self.model.load('../output/M0256/M0256-999')

        self.train_x = self.model.predict(self.train_x).reshape((-1, 3, 256, 3, 1))
        self.test_x = self.model.predict(self.test_x).reshape((-1, 3, 256, 3, 1))

        del self.model

        model_name = 'C0256-adagrad-huber-50steps-2000bs'

        self.conv = ConvDense(model_name=model_name)
        self.conv.build(n_models=3, n_neurons_per_layer=256)

        self.conv.optimize(x=self.train_x, y=self.train_y, x_test=self.train_x[38000:,:], y_test=self.train_y[38000:,:],
                           learning_rate=.25, steps=50, batch_size=1000)

        y = self.conv.predict(self.test_x)

        pd.DataFrame({'ImageId': list(range(1, len(y) + 1)), 'Label': y}).to_csv(
            '../output/{}.csv'.format(model_name), sep=',', index=False)
