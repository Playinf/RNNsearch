# ops.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import tensorflow as tf


def scan(fn, seqs, outputs_info, non_seq=None, n_steps=None):
    if n_steps is not None:
        time_steps = n_steps
    else:
        time_steps = tf.shape(seqs[0])[0]

    if not non_seq:
        non_seq = []

    # create tensor array
    input_ta = []

    for seq in seqs:
        array = tf.TensorArray(seq.dtype, time_steps)
        array = array.unpack(seq)
        input_ta.append(array)

    output_ta = []
    states = []

    for info in outputs_info:
        if info is None:
            array = tf.TensorArray(tf.float32, time_steps)
        else:
            array = tf.TensorArray(info.dtype, time_steps)
            states.append(info)

        output_ta.append(array)

    cond = lambda t, *_: t < time_steps

    time = tf.constant(0, tf.int32)

    def body(time, output_ta, states):
        seq = [ta.read(time) for ta in input_ta]

        outputs = fn(*(seq + states + non_seq))

        output_ta = [ta.write(time, v) for ta, v in zip(output_ta, outputs)]
        newstates = []

        for val, info in zip(outputs, outputs_info):
            if info is not None:
                newstates.append(val)

        return (time + 1, output_ta, newstates)

    loop_vars = [time, output_ta, states]
    t, output_ta, states = tf.while_loop(cond, body, loop_vars,
                                         swap_memory=True)

    outputs = [ta.pack() for ta in output_ta]

    if len(outputs) == 1:
        outputs = outputs[0]

    return outputs


def function(nodes, outputs, updates=None):
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    fetches = list(outputs)

    if updates:
        fetches += [updates]

    def func(*inputs, **opt):
        if len(inputs) != len(nodes):
            raise ValueError("inputs do not match placeholders")
        feed_dict = {}

        for placeholder, input in zip(nodes, inputs):
            feed_dict[placeholder] = input

        if "session" not in opt:
            session = None
        else:
            session = opt["session"]

        if not session:
            session = tf.get_default_session()

        results = session.run(fetches, feed_dict=feed_dict)

        if len(outputs) == 1:
            return results[0]
        else:
            if updates:
                return results[:-1]
            return results

    return func
