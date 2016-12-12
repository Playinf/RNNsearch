# multirnn.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import tensorflow as tf

from tensorflow.python.util import nest


class multirnn:

    def __init__(self, cells, residual=None):
        if not isinstance(cells, (list, tuple)):
            cells = [cells]

        if residual and not isinstance(residual, int):
            raise ValueError("residual must be None or int")

        input_size = cells[0].size[0]
        hidden_size = [cell.size[1] for cell in cells]
        flatten_size = nest.flatten(hidden_size)

        params = []

        for cell in cells:
            params.extend(cell.parameter)

        def make_state(batch, val=0):
            state = []
            for size in flatten_size:
                state.append(tf.zeros([batch, size]))

            return state

        self.cell = cells
        self.size = [input_size, hidden_size]
        self.state_size = flatten_size
        self.residual = residual
        self.parameter = params
        self.make_state = make_state

    def __call__(self, inputs, state):
        residual = self.residual
        num_cells = len(self.cell)
        new_state = []

        if isinstance(inputs, (list, tuple)):
            inputs, other_inputs = inputs

            if len(other_inputs) != num_cells:
                raise ValueError("unmatched input number and layer number")
        else:
            other_inputs = None

        output = inputs
        state = nest.pack_sequence_as(self.size[1], state)

        for i in range(num_cells):
            prev_state = state[i]

            if other_inputs:
                output = [output] + list(other_inputs[i])

            new_output, next_state = self.cell[i](output, prev_state)
            new_state.append(next_state)

            if residual and i >= residual:
                if other_inputs:
                    output = output[0]

                output = output + new_output
            else:
                 output = new_output

        new_state = nest.flatten(new_state)

        return output, new_state
