from model import Dense
import numpy as np
import pandas as pd
#
# Loading train dataset
#
test = pd.read_table('../input/mnist/test.csv', sep=',')
test_x = test.as_matrix().astype(float)

#
# Predict
#

abstraction_layer = Dense()
abstraction_layer.load('../output/model/M0000/M0000-999')

y_hat = abstraction_layer.predict__(test_x)

print(y_hat.shape)

for index, model in enumerate(y_hat):
    r = np.argmax(model, 1)
    print(r.shape)
    result = pd.DataFrame({'ImageId': list(range(1, len(r) + 1)), 'Label': r})
    result.to_csv('../output/benchmark/mnist/M0000-sgd-1000steps-{}.csv'.format(['sigmoid', 'tanh', 'relu'][index]), sep=',', index=False)