# gru.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import tensorflow as tf

from config import config
from utils import get_or_default
from linear import linear, linear_config
from feedforward import feedforward, feedforward_config


class gru_config(config):
    """
    * dtype: dtype, default tf.float32
    * scope: str, default "gru"
    * concat: bool, True to concat weight matrices
    * activation: activation function, default tanh
    * gates: feedforward_config, config behavior of gates
    * reset_gate: feedforward_config, config behavior of reset gate
    * update_gate: feedforward_config, config behavior of update gate
    * candidate: linear_config, config behavior of candidate transform
    """

    def __init__(self, **kwargs):
        self.dtype = get_or_default(kwargs, "dtype", tf.float32)
        self.scope = get_or_default(kwargs, "scope", "gru")
        self.concat = get_or_default(kwargs, "concat", False)
        self.activation = get_or_default(kwargs, "activation", tf.tanh)
        self.gates = feedforward_config(dtype=self.dtype, scope="gates")
        self.reset_gate = feedforward_config(dtype=self.dtype,
                                             scope="reset_gate")
        self.update_gate = feedforward_config(dtype=self.dtype,
                                              scope="update_gate")
        self.candidate = linear_config(dtype=self.dtype, scope="candidate")


# gated recurrent unit
class gru:

    def __init__(self, input_size, output_size, config=gru_config()):
        scope = config.scope
        concat = config.concat
        activation = config.activation

        if not isinstance(input_size, (list, tuple)):
            input_size = [input_size]

        modules = []

        # config scope
        with tf.variable_scope(scope):
            if not concat:
                isize = input_size + [output_size]
                osize = output_size
                rgate = feedforward(isize, osize, config.reset_gate)
                ugate = feedforward(isize, osize, config.update_gate)
                trans = linear(isize, osize, config.candidate)

                modules.append(rgate)
                modules.append(ugate)
                modules.append(trans)
            else:
                isize = input_size + [output_size]
                osize = output_size
                gates = feedforward(isize, 2 * osize, config.gates)
                trans = linear(isize, osize, config.candidate)

                modules.append(gates)
                modules.append(trans)

        params = []

        for m in modules:
            params.extend(m.parameter)

        def forward(x, h):
            if not isinstance(x, (list, tuple)):
                x = [x]

            if not concat:
                reset_gate = modules[0]
                update_gate = modules[1]
                transform = modules[2]
                r = reset_gate(x + [h])
                u = update_gate(x + [h])
                c = activation(transform(x + [r * h]))
            else:
                gates = modules[0]
                transform = modules[1]
                r_u = gates(x + [h])
                r, u = tf.split(r_u.get_shape().ndims - 1, 2, r_u)
                c = activation(transform(x + [r * h]))

            y = (1.0 - u) * h + u * c

            return y, y

        def zero_state(batch):
            return tf.zeros([batch, output_size])

        self.name = scope
        self.size = [input_size, output_size]
        self.config = config
        self.forward = forward
        self.parameter = params
        self.zero_state = zero_state

    def __call__(self, inputs, state):
        return self.forward(inputs, state)
