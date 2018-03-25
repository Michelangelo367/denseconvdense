import unittest
import pandas as pd

from model import DenseConvDense


class TestDenseConvDense(unittest.TestCase):

    def setUp(self):
        train = pd.read_table('../input/mnist/train.csv', sep=',')

        self.train_x = train.iloc[:, 1:].as_matrix()
        self.train_y = pd.get_dummies(train.iloc[:, 0]).as_matrix()

        #self.test = pd.read_table('../input/test.csv', sep=',')

    def test_model(self):

        self.model = DenseConvDense(n_input_features=self.train_x.shape[1], n_outputs=self.train_y.shape[1],
                                    abstraction_activation_functions=('sigmoid', 'tanh', 'relu'),
                                    n_hidden_nodes=100, keep_probability=0.5, initialization='RBM')

        self.model.build()

        self.model.optimize(self.train_x, self.train_y)
