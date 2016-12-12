# maxout.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import tensorflow as tf

from linear import linear
from utils import get_or_default
from config import config, option


class maxout_config(config):
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
        self.scope = get_or_default(kwargs, "scope", "maxout")
        self.concat = get_or_default(kwargs, "concat", False)
        self.multibias = get_or_default(kwargs, "multibias", False)
        self.bias = option(use=True, initializer=tf.zeros_initializer)
        self.weight = option(output_major=False,
                             initializer=tf.uniform_unit_scaling_initializer())


# maxout unit
# input_size: dimension of x
# output_size: dimension of y
class maxout:

    def __init__(self, input_size, output_size, maxpart=2,
                 config=maxout_config()):
        scope = config.scope
        k = maxpart

        transform = linear(input_size, output_size * k, config)

        def forward(inputs):
            z = transform(inputs)
            dim = z.get_shape().ndims
            shape = tf.shape(z)
            shape_list = [shape[i] for i in range(dim)]
            shape_list[-1] = shape_list[-1] / k
            shape_list += [k]
            z = tf.reshape(z, shape_list)
            y = tf.reduce_max(z, dim)

            return y

        self.name = scope
        self.config = config
        self.forward = forward
        self.parameter = transform.parameter

    def __call__(self, inputs):
        return self.forward(inputs)
