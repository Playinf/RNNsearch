# embedding.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import tensorflow as tf

from utils import get_or_default
from config import config, option


class embedding_config(config):
    """
    * dtype: dtype, default tf.float32
    * scope: str, default "embedding"
    * initializer: function, use to initialize embedding
    * bias: config.option, set bias.use=True to use bias, set bias.initializer
            to select initializer
    """

    def __init__(self, **kwargs):
        initializer = tf.uniform_unit_scaling_initializer()
        self.dtype = get_or_default(kwargs, "dtype", tf.float32)
        self.scope = get_or_default(kwargs, "scope", "embedding")
        self.initializer = get_or_default(kwargs, "initializer", initializer)
        self.bias = option(use=True, initializer=tf.zeros_initializer)


# embedding
# representing embedding layer
# num: number of entries
# dim: vector dimension
class embedding:

    def __init__(self, num, dim, config):
        dtype = config.dtype
        scope = config.scope
        initializer = config.initializer
        use_bias, b_initializer = tuple(config.bias)

        params = []

        # allocate embedding on CPU to save memory
        with tf.variable_scope(scope), tf.device("/cpu:0"):
            init_val = initializer([num, dim], dtype=dtype)
            emb = tf.Variable(init_val, name="embedding")
            params.append(emb)

            if use_bias:
                init_val = b_initializer([dim,], dtype=dtype)
                bias = tf.Variable(init_val, name="bias")
                params.append(bias)

        def forward(indices):
            with tf.device("/cpu:0"):
                values = tf.nn.embedding_lookup(emb, indices)

                if use_bias:
                    values = values + bias

            return values

        self.scope = scope
        self.config = config
        self.forward = forward
        self.parameter = params
        self.embedding = emb

    def __call__(self, indices):
        return self.forward(indices)
