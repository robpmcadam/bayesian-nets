"""
file for custom Keras layers for bayesian models
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow_probability as tfp
from typing import (
    Union,
    Literal,
    Callable,
)

from bayesian_modeling.utils import (
    activation_map,
    add_dims_to_inputs,
    init_trainable_bias,
    init_trainable_weights,
    kl_divergence,
)


tfd = tfp.distributions


class DenseVariational(Layer):
    """
    base variational inference Keras layer
    """

    def __init__(
        self,
        units: int,
        n_observatons: int = 1,
        bias: bool = False,
        activation: Union[
            Literal[
                "leaky_relu", "relu", "sigmoid", "softmax", "softplus", "swish", "tanh"
            ],
            Callable,
        ] = None,
        prior_dist: tfd.Distribution = tfd.StudentT,
        variational_posterior_dist: tfd.Distribution = tfd.Normal,
        prior_params: dict = {"df": 2, "loc": 0, "scale": 1},
        variational_posterior_params: dict = {"loc": 0, "scale": 0.5},
    ):
        super().__init__()
        self.units = units
        self.n_observations = n_observatons
        self.bias = init_trainable_bias(units, bias)
        self.activation = activation_map(activation)
        self.prior_dist = prior_dist
        self.prior_params = prior_params
        self.variational_posterior_dist = variational_posterior_dist
        self.variational_posterior_params = variational_posterior_params
        self.variational_posterior = None

    def build(self, input_shape):
        super().build(input_shape)
        target_shape = [
            input_shape[-1],
            self.units,
        ]

        if not self.prior:
            self.prior = self.prior_dist(
                **init_trainable_weights(self.prior_params, target_shape, trainable=False)
            )

        if not self.variational_posterior:
            self.variational_posterior = self.variational_posterior_dist(
                **init_trainable_weights(
                    self.variational_posterior_params, target_shape
                )
            )
            for variable in self.variational_posterior.trainable_variables:
                self._trainable_weights.append(variable)

    def forward(self, inputs: tf.Tensor, weights):
        return self.activation(tf.einsum("dni,diu->dnu", inputs, weights) + self.bias)

    def call(self, inputs: tf.Tensor, n_posterior_samples=50):

        weights = self.variational_posterior.sample(n_posterior_samples)

        complexity = (
            tf.reduce_sum(
                kl_divergence(self.variational_posterior, self.prior, q_samps=weights)
            )
            / self.n_observations
        )

        self.add_loss(complexity)

        expanded_inputs = add_dims_to_inputs(inputs, n_posterior_samples)
        return self.forward(expanded_inputs, weights)
