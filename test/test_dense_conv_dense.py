import unittest
import pandas as pd
import numpy as np

from model import DenseConvDense


class TestDenseConvDense(unittest.TestCase):

    def setUp(self):

        train = pd.read_table('../input/mnist/train.csv', sep=',')

        self.train_x = train.iloc[:, 1:].as_matrix()

        self.train_y = pd.get_dummies(train.iloc[:, 0]).as_matrix()

        #self.test = pd.read_table('../input/test.csv', sep=',')

    def test_model(self):

        self.model = DenseConvDense()

        self.model.build(n_input_features=self.train_x.shape[1], n_outputs=self.train_y.shape[1],
                         abstraction_activation_functions=('sigmoid', 'tanh', 'relu'),
                         n_hidden_layers=3, optimizer_algorithms=('sgd', 'sgd', 'sgd'),
                         n_hidden_nodes=100, keep_probability=0.5, initialization='RBM',
                         batch_normalization=True)

        self.model.optimize(self.train_x, self.train_y)

    def test_load(self):

        self.model = DenseConvDense()

        self.model.load('../output/M0010/M0000-110')

        x = np.random.rand(1,784)

        print(np.array_str(np.array(self.model.predict(x)), precision=2))
