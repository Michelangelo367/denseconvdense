### New Deep Learning Architecture

It aims at defining a new multidimensional representation of tabular data to apply convolutional layers in order to
create better models.

### Experiments

| **Code** | **Dataset** | **Model Description** | Batch Size | Iterations | Learning Rate | Batch Normalization | Dropout Hidden | Dropout Input |
|----------|:-----------:|-----------------------|:----------:|:----------:|:-------------:|:-------------------:|:--------------:|:-------------:|
| 0001 | MNIST from Kaggle | 3 hidden layers with 100 neurons each. 3 models using sigmoid, tanh, and relu functions. Output with identity activation function. | 1000 | 1000 | 1e-5 | No | 0.5 | 0.5 |