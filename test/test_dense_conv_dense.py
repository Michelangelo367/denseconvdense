import unittest
import pandas as pd
import numpy as np

from model import Dense, ConvDense


class TestDenseConvDense(unittest.TestCase):

    def setUp(self):

        train = pd.read_table('../input/mnist/train.csv', sep=',')

        self.train_x = train.iloc[:40000, 1:].as_matrix()
        self.train_y = pd.get_dummies(train.iloc[:40000, 0]).as_matrix().astype(int)

        self.test_x = train.iloc[40000:, 1:].as_matrix()
        self.test_y = pd.get_dummies(train.iloc[40000:, 0]).as_matrix().astype(int)


    def test_model(self):

        self.model = Dense(model_name='M0001')

        self.model.build(n_input_features=self.train_x.shape[1], n_outputs=self.train_y.shape[1],
                         abstraction_activation_functions=('sigmoid', 'tanh', 'relu'),
                         n_hidden_layers=3, optimizer_algorithms=('sgd', 'sgd', 'sgd'),
                         n_hidden_nodes=100, keep_probability=0.5,
                         initialization='RBM', batch_normalization=True)

        self.model.optimize(self.train_x, self.train_y, learning_rate=1e-3)

    def test_load(self):

        self.model = Dense()

        self.model.load('../output/model/M0000/M0000-999')

        transformed_x = self.model.predict(self.train_x)

        self.assertEqual(transformed_x.shape, (40000, 3, 100, 3))

    def test_conv_dense(self):

        self.model = Dense()
        self.model.load('../output/model/M0000/M0000-999')

        self.train_x = self.model.predict(self.train_x).reshape((-1, 3, 100, 3, 1))
        self.test_x = self.model.predict(self.test_x).reshape((-1, 3, 100, 3, 1))

        self.conv = ConvDense(model_name='C0000-ftrl')
        self.conv.build()

        self.conv.optimize(x=self.train_x, y=self.train_y, x_test=self.test_x, y_test=self.test_y,
                           learning_rate=.8, steps=50, batch_size=1000)

        y = self.conv.predict(self.test_x)

        print(np.mean(np.equal(np.argmax(y, 1), np.argmax(self.test_y, 1))))