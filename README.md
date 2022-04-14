# bayesian-nets
A Bayesian TensorFlow library powering variational Bayesian neural networks. 

This API provides a high level of abstraction on top of Keras that enables parsimonious, 
configurable, and powerful models.

## API overview

#### Layers
The `DenseVariational` class in `bayesian_modeling/layers.py` inherits from the base Keras `Layer
` class and is the basic building block for the Bayesian models in this library.

The layer serves as a container for the prior on the parameters (by default a unit 2-dof Student-t
 ball
 in order to encode sparsity) and the trainable variational posterior, computes stochastic
  estimates of the complexity 
  (<img src="https://render.githubusercontent.com/render/math?math={\color{gray}D_{KL}[q(\theta) || p(\theta)]}">), 
  performs dense matrix multiplication, and applies an activation function.

#### Models

The `ModelVI` class in `bayesian_modeling/models.py` is a subclass of the base Keras `Model
` class. It wraps variational inference specific logic into the `fit` method; namely, the number
 of observations in the dataset is assigned to the variational layers in order to balance the
  complexity and accuracy terms that comprise the variational free energy during mini-batching.
  

The `BayesianNet` class provides a configurable variational Bayesian neural network. Some
 configurables include batch-normalization, choice of distribution head (Gaussian, Bernoulli, and
  Multinomial), number of hidden layers, etc. This is intended to allow users to create a model
   ready for training with a single line of code.
   
## Requirements
```pip install -r requirements.txt```

## Installing
```git clone https://github.com/robpmcadam/bayesian-nets.git```