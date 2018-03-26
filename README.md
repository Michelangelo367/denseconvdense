### New Deep Learning Architecture

It aims at defining a new multidimensional representation of tabular data to apply convolutional layers in order to
create better models.

### Experiments

#### Abstraction Layer

| **ID**   | **Dataset**       | **Hidden Layers** | **Neurons per Layer** | **Models**          | **Output Function** | **Iterations** | **Batch Size** |Learning Rate | Batch Normalization | Dropout Input | Dropout Hidden | Batch Shuffle | L1  | L2  | Data Normalization |
|:--------:|:-----------------:|------------------:|----------------------:|:-------------------:|:-------------------:|---------------:|:--------------:|:------------:|:-------------------:|--------------:|:--------------:|:-------------:|:---:|:---:|:------------------:|
| M0001    | MNIST from Kaggle | 3                 | 100                   | sigmoid, tanh, relu | identity            | 1000           | 1000           | 1e-5         | Yes                 | 0.5           | 0.5            | Yes           | No  | No  | No                 |

### References

1. [Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)
2. [Why Does Unsupervised Pre-training Help Deep Learning?](http://www.jmlr.org/papers/volume11/erhan10a/erhan10a.pdf)

### Ideas

Test different configurations when merging neurons values to apply a convolution network:
1. Merge at the same matrix layers of the same model
2. Merge at the same matrix correspondent layers of different models
