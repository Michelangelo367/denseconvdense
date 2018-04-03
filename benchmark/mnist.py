from model import Dense, ConvDense

import pandas as pd
#
# Loading train dataset
#
train = pd.read_table('../input/mnist/train.csv', sep=',')
train_x = train.iloc[:, 1:].as_matrix().astype(float)
train_y = pd.get_dummies(train.iloc[:, 0]).as_matrix().astype(int)

#
# Training
#

abstraction_layer = Dense()
abstraction_layer.load('../output/model/M0000/M0000-999')

conv_dense = ConvDense(abstraction_layer=abstraction_layer,
                       model_name='C0000-banchmark-mnist', summaries_dir=None)
conv_dense.build()

conv_dense.optimize(train_x, train_y,
                    x_test=train_x[40000:,:], y_test=train_y[40000:],
                    learning_rate=.5, steps=1000)
#
# Predicting
#
test = pd.read_table('../input/mnist/test.csv', sep=',')
test_x = test.as_matrix().astype(float)
y_hat = conv_dense.predict(test_x)
#
# Exporting
#

result = pd.DataFrame({'ImageId': list(range(1, len(y_hat) + 1)), 'Label': y_hat})
result.to_csv('../output/banchmark/mnist/M0000-C0000-modified-adagrad-1000step.csv', sep=',', index=False)