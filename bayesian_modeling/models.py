"""
file for custom Keras models for bayesian modeling
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, Optimizer
import tensorflow_probability as tfp
from typing import Literal
from bayesian_modeling.utils import nll_loss
from bayesian_modeling.layers import DenseVariational

tfd = tfp.distributions


class ModelVI(Model):
    """
    base variational inference Keras model
    """

    def fit(self, x, y, **fit_kwargs):
        self.set_n_observations(len(x))
        return super().fit(x, y, **fit_kwargs)

    def set_n_observations(self, n_observations: int):
        for layer in self.layers:
            if isinstance(layer, DenseVariational):
                layer.n_observations = n_observations


class BayesianNet(ModelVI):
    """
    Bayesian MLP with a probabilistic head
    """

    def __init__(
        self,
        name="BayesianNet",
        model_type: Literal["gaussian", "bernoulli", "multinomial"] = "gaussian",
        optimizer: Optimizer = Adam,
        learning_rate: float = 1e-3,
        n_outputs=1,
        n_hidden_layers=2,
        neurons_per_layer=128,
        use_batchnorm=True,
        **kwargs
    ):
        """
        params:
            name: name of model
            model_type: distribution head
            optimizer: optimizer
            learning_rate: optimizer learning rate
            n_outputs: number of model outputs
            n_hidden_layers: number of hidden layers
            neurons_per_layer: number of neurons per layer
            use_batchnorm: use batch normalization between layers
            kwargs: keyword args passed to model
        """
        super().__init__(name=name, **kwargs)
        self.model_type = model_type
        self.optimizer = optimizer(learning_rate)
        self.loss = nll_loss
        self.base_layers = []
        if use_batchnorm:
            self.base_layers.append(BatchNormalization())
            for _ in range(n_hidden_layers):
                self.base_layers.extend(
                    [
                        DenseVariational(neurons_per_layer, activation="swish"),
                        BatchNormalization(),
                    ]
                )
        else:
            for _ in range(n_hidden_layers):
                self.base_layers.extend(
                    [DenseVariational(neurons_per_layer, activation="swish")]
                )

        if model_type == "gaussian":
            self.head_layers = [
                DenseVariational(n_outputs, bias=True),
                DenseVariational(n_outputs, bias=True, activation="softplus"),
            ]
        elif model_type == "bernoulli":
            self.head_layers = [DenseVariational(n_outputs, bias=True, activation="sigmoid")]
        else:
            self.head_layers = [DenseVariational(n_outputs, bias=True, activation="softmax")]

    def call(self, inputs, training=None, n_samples=1):
        dist = self._get_dist(inputs)
        if training:
            return dist
        else:
            return dist.sample(n_samples)

    def _get_dist(self, inputs):
        x = inputs
        for layer in self.base_layers:
            x = layer(x)
        if self.model_type == "gaussian":
            return tfd.Normal(self.head_layers[0](x), 0.001 + self.head_layers[1](x))
        elif self.model_type == "bernoulli":
            return tfd.Bernoulli(self.head_layers[0](x))
        else:
            return tfd.Categorical(self.head_layers[0](x))
