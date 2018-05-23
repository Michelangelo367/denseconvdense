import unittest
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from model import *


class TestDenseConvDense(unittest.TestCase):

    def setUp(self):
        self.dataset = pd.read_csv('../input/oralinfection/dataset.csv')

    def test_dense_conv_oral_infection(self):

        kfold = KFold(10)

        sum_auc, sum_acc = 0, 0

        for i, (train_index, test_index) in enumerate(kfold.split(self.dataset)):

            scaler = MinMaxScaler()

            train_x = scaler.fit_transform(self.dataset.iloc[train_index,:-1].as_matrix())
            train_y = self.dataset.iloc[train_index, -1]

            test_x = scaler.transform(self.dataset.iloc[test_index, :-1].as_matrix())
            test_y = self.dataset.iloc[test_index, -1].astype(float)

            model_name = 'DCD_0001_INFECTION_OPT_HUBER_ANN_32_LR_1E-3_BS_025_FOLD_{}'.format(i if i > 9 else '0' + str(i))

            self.model = DenseConvDense(model_name=model_name)

            self.model.build(train_x.shape[1], 1, abstraction_n_neurons_per_hidden_layer=32)

            self.model.optimize(x=train_x, y=train_y, x_test=test_x, y_test=test_y,
                               learning_rate=1e-3, steps=100, batch_size=25)

            y_hat = self.model.predict(test_x)

            y_hat[y_hat >= .5] = 1.
            y_hat[y_hat < .5] = 0.

            auc, acc = roc_auc_score(test_y, y_hat), accuracy_score(test_y, y_hat)

            sum_auc += auc
            sum_acc += acc

            print('Fold {}: ACC = {}, AUC = {}'.format(i + 1, acc, auc))

        print('=================================================================')
        print('MEAN: ACC = {}, AUC = {}'.format(sum_acc / 10., sum_auc / 10.))

    def test_dense_infection(self):

        kfold = KFold(10)

        activation_functions = ('sigmoid', 'tanh', 'relu')

        sum_auc, sum_acc = {}, {}

        for af in activation_functions:
            sum_auc[af], sum_acc[af] = 0., 0.

        for i, (train_index, test_index) in enumerate(kfold.split(self.dataset)):

            scaler = MinMaxScaler()

            train_x = scaler.fit_transform(self.dataset.iloc[train_index, :-1].as_matrix())

            y_tmp = self.dataset.iloc[train_index, -1].as_matrix()
            train_y = np.array([y_tmp, 1 - y_tmp], dtype=np.float32).T

            test_x = scaler.transform(self.dataset.iloc[test_index, :-1].as_matrix())

            y_tmp = self.dataset.iloc[test_index, -1].as_matrix()
            test_y = np.array([y_tmp, 1 - y_tmp], dtype=np.float32).T

            del y_tmp

            self.model = Dense(model_name='DENSE_INFECTION_HL_3_HN_128_LOG_BS_ALL_W_60_40_FTRL_FOLD_{}'.format(i + 1))

            self.model.build(n_features=train_x.shape[1], n_outputs=2,
                             abstraction_activation_functions=activation_functions,
                             n_hidden_layers=3, optimizer_algorithms=('ftrl', 'ftrl', 'ftrl'),
                             n_hidden_nodes=128, keep_probability=0.5,
                             initialization='RBM', batch_normalization=True)

            self.model.optimize(train_x, train_y, test_x, test_y,
                                learning_rate=1e-3, batch_size=train_x.shape[1], steps=1000, weights=[.60, .40])

            y_hats = self.model.predict__(test_x)

            for index, y_hat in enumerate(y_hats):

                af = activation_functions[index]

                y_hat[y_hat >= .5] = 1.

                y_hat[y_hat < .5] = 0.

                auc, acc = roc_auc_score(test_y, y_hat), accuracy_score(test_y, y_hat)

                sum_auc[af] += auc
                sum_acc[af] += acc

                print('{} Fold {}: ACC = {}, AUC = {}'.format(af, i + 1, acc, auc))

        for af in sum_acc:
            print('=================================================================')
            print('MEAN {}: ACC = {}, AUC = {}'.format(af, sum_acc[af]/ 10., sum_auc[af] / 10.))

    def test_pre_trained_dense_conv_dense_infection(self):

        kfold = KFold(10)

        sum_auc, sum_acc = 0,0

        for i, (train_index, test_index) in enumerate(kfold.split(self.dataset)):

            scaler = MinMaxScaler()

            train_x = scaler.fit_transform(self.dataset.iloc[train_index, :-1].as_matrix())
            train_y = self.dataset.iloc[train_index, -1].as_matrix().reshape((-1, 1))

            test_x = scaler.transform(self.dataset.iloc[test_index, :-1].as_matrix())
            test_y = self.dataset.iloc[test_index, -1].astype(float).as_matrix().reshape((-1, 1))

            self.model = Dense()

            name = 'DENSE_INFECTION_HL_3_HN_128_SIG_TAN_REL_LOGLOSS_BS_ALL_FOLD_{}'.format(i + 1)

            self.model.load('../output/{0}/{0}-99'.format(name))

            train_x = self.model.predict(train_x)
            test_x = self.model.predict(test_x)

            del self.model

            model_name = 'CD_INFECTION_LR_25_OPT_ADADELTA_BS_ALL_LOSS_LOG1000_MM_FOLD_{}'.format(i+1)

            self.conv = ConvDense(model_name=model_name)
            self.conv.build(n_outputs=1, n_models=3, n_neurons_per_layer=128, optimizer_algorithm='adadelta')

            self.conv.optimize(x=train_x, y=train_y, x_test=test_x, y_test=test_y,
                               learning_rate=.25, steps=50, batch_size=train_x.shape[0])

            y_hat = self.conv.predict(test_x)

            y_hat[y_hat >= .5] = 1.

            y_hat[y_hat < .5] = 0.

            auc, acc = roc_auc_score(test_y, y_hat), accuracy_score(test_y, y_hat)

            sum_auc += auc
            sum_acc += acc

            print('Fold {}: ACC = {}, AUC = {}'.format(i + 1, acc, auc))

        print('=================================================================')
        print('MEAN: ACC = {}, AUC = {}'.format(sum_acc/ 10., sum_auc / 10.))




