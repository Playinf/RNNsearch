# multirnn.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import tensorflow as tf


class dropout_rnn:

    def __init__(self, cell, keep_prob=1.0):
        if keep_prob > 1.0 or keep_prob <= 0.0:
            raise ValueError("keep_prob must in range (0, 1.0]")

        self.cell = cell
        self.enable = True
        self.keep_prob = keep_prob
        self.name = cell.name
        self.size = cell.size
        self.config = cell.config
        self.parameter = cell.parameter
        self.zero_state = cell.zero_state

    def __call__(self, inputs, state):
        output, new_state = self.cell(inputs, state)
        if self.enable and self.keep_prob < 1.0:
            output = tf.nn.dropout(output, self.keep_prob)

        return output, new_state
