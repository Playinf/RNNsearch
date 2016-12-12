# linear.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import tensorflow as tf

from utils import get_or_default
from config import config, option


class linear_config(config):
    """
    * dtype: dtype, default tf.float32
    * scope: str, default "linear"
    * concat: bool, True to concate weights, False to use seperate weights
    * multibias: bool, True to use bias per input, only works when
    *            concat = False
    * bias: config.option, set bias.use=True to use bias, set bias.initializer
            to set initializer
    * weight: config.option, output_major=True to change weigth matrix
              to [output_size, input_size], weight.initializer to set
              initializer
    """

    def __init__(self, **kwargs):
        self.dtype = get_or_default(kwargs, "dtype", tf.float32)
        self.scope = get_or_default(kwargs, "scope", "linear")
        self.concat = get_or_default(kwargs, "concat", False)
        self.multibias = get_or_default(kwargs, "multibias", False)
        self.bias = option(use=True, initializer=tf.zeros_initializer)
        self.weight = option(output_major=False,
                             initializer=tf.uniform_unit_scaling_initializer())


def nd_matmul(x, y):
    ndim = x.get_shape().ndims

    if ndim == 2:
        return tf.matmul(x, y)
    else:
        xshape = tf.shape(x)

        x = tf.reshape(x, [-1, xshape[ndim - 1]])
        z = tf.matmul(x, y)
        newshape = []

        for i in range(ndim - 1):
            newshape.append(xshape[i])

        newshape.append(-1)

        return tf.reshape(z, newshape)


# linear map
# input_size: dimension of x
# output_size: dimension of y
class linear:

    def __init__(self, input_size, output_size, config=linear_config()):
        dtype = config.dtype
        scope = config.scope
        concat = config.concat
        multibias = config.multibias
        use_bias, b_initializer = tuple(config.bias)
        output_major, w_initializer = tuple(config.weight)

        if not isinstance(input_size, (list, tuple)):
            input_size = [input_size]

        params = []
        weights = []
        biases = []

        with tf.variable_scope(scope):
            if concat:
                input_size = sum(input_size)

                if output_major:
                    shape = [output_size, input_size]
                else:
                    shape = [input_size, output_size]

                init_val = w_initializer(shape, dtype=dtype)
                weight = tf.Variable(init_val, name="weight")
                params.append(weight)
                weights.append(weight)
            else:
                for i in range(len(input_size)):
                    if output_major:
                        shape = [output_size, input_size[i]]
                    else:
                        shape = [input_size[i], output_size]

                    init_val = w_initializer(shape, dtype=dtype)
                    weight = tf.Variable(init_val, name="weight")
                    params.append(weight)
                    weights.append(weight)

            if use_bias:
                shape = [output_size]
                if not concat and multibias:
                    for i in range(len(input_size)):
                        init_val = b_initializer(shape, dtype=dtype)
                        bias = tf.Variable(init_val, name="bias")
                        params.append(bias)
                        biases.append(bias)
                else:
                    init_val = b_initializer(shape, dtype=dtype)
                    bias = tf.Variable(init_val, name="bias")
                    params.append(bias)
                    biases.append(bias)

        def forward(x):
            if not isinstance(x, (list, tuple)):
                x = [x]

            if len(x) != len(input_size):
                raise ValueError("unmatched inputs and weights")

            outs = []
            n = len(x)

            if concat:
                if len(x) == 1:
                    x = x[0]
                else:
                    x = tf.concat(x.get_shape().ndims - 1, x)

                if output_major:
                    outs.append(nd_matmul(x, tf.transpose(weights[0])))
                else:
                    outs.append(nd_matmul(x, weights[0]))

                if use_bias:
                    outs.append(biases[0])
            else:
                for i in range(n):
                    if output_major:
                        weight = tf.transpose(weights[i])
                        outs.append(nd_matmul(x[i], weight))
                    else:
                        outs.append(nd_matmul(x[i], weights[i]))

                    if use_bias and multibias:
                        outs.append(biases[i])

                if use_bias and not multibias:
                    outs.append(biases[0])

            y = reduce(tf.add, outs)

            return y

        self.name = scope
        self.config = config
        self.parameter = params
        self.forward = forward

    def __call__(self, x):
        return self.forward(x)
