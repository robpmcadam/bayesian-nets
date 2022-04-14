"""
file for custom functions
"""
from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp
from toolz.dicttoolz import itemmap
from typing import (
    Union,
    Literal,
    Callable,
)

tfb = tfp.bijectors
tfd = tfp.distributions


def add_dims_to_inputs(inputs, n_samples):
    if len(inputs.shape) < 3:
        return tf.repeat(tf.expand_dims(inputs, 0), n_samples, axis=0)
    else:
        return inputs


def kl_divergence(
    q: tfd.Distribution,
    p: tfd.Distribution,
    n_samps: int = 50,
    q_samps: tf.Tensor = None,
) -> tf.Tensor:
    if q_samps is None:
        q_samps = q.sample(n_samps)
    return tf.reduce_mean(q.log_prob(q_samps) - p.log_prob(q_samps), axis=0)


def init_trainable_bias(
    units: int,
    use_bias: bool,
):
    if use_bias:
        return (tf.Variable(tf.zeros([units]) + tfd.Normal(0, 0.01).sample(units)),)
    else:
        return tf.zeros([units])


def init_trainable_weights(
    params: dict, target_shape: list[int], trainable: bool = True
):
    def _init_param(param):
        key, val = param
        if key == "scale":
            return key, tfp.util.TransformedVariable(
                tf.Variable(
                    tf.ones(target_shape) * val
                    + tfd.Normal(0, 0.01).sample(target_shape),
                    trainable=trainable,
                ),
                bijector=tfb.Softplus(),
            )
        else:
            return key, tf.Variable(
                tf.ones(target_shape) * val + tfd.Normal(0, 0.01).sample(target_shape),
                trainable=trainable,
            )

    return itemmap(_init_param, params)


def activation_map(
    activation: Union[
        Literal[
            "leaky_relu", "relu", "sigmoid", "softmax", "softplus", "swish", "tanh"
        ],
        Callable,
    ]
) -> Callable:
    if not activation:
        return tf.identity
    elif callable(activation):
        return activation
    elif activation == "leaky_relu":
        return tf.nn.leaky_relu
    elif activation == "relu":
        return tf.nn.relu
    elif activation == "sigmoid":
        return tf.nn.sigmoid
    elif activation == "softmax":
        return tf.nn.softmax
    elif activation == "softplus":
        return tf.nn.softplus
    elif activation == "swish":
        return tf.nn.swish
    elif activation == "tanh":
        return tf.nn.tanh
    else:
        raise (
            "Accepted activations: leaky_relu, relu, sigmoid, softmax, softplus, swish, tanh, "
            "or tf activation fn"
        )


def nll_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return -tf.reduce_mean(y_pred.log_prob(y_true))
