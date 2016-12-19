# rnnsearch.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy as np
import tensorflow as tf

from utils import function
from search import beam, select_nbest
from tensorflow.python.ops.rnn import _rnn_step as rnn_step
from tensorflow.python.ops.rnn_cell import _linear as linear


def maxout(inputs, size, maxpart, use_bias=True, scope=None):
    with tf.variable_scope(scope or "maxout"):
        candidate = linear(inputs, size * maxpart, use_bias)
        value = tf.reshape(candidate, [-1, size, maxpart])
        output = tf.reduce_max(value, 2)

    return output


class gru_cell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, output_size):
        self.size = output_size

    def __call__(self, inputs, state, scope=None):
        output_size = self.size

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        with tf.variable_scope(scope or "gru_cell"):
            x = inputs + [state]
            with tf.variable_scope("reset-gate"):
                r = tf.sigmoid(linear(x, output_size, False))
            with tf.variable_scope("update-gate"):
                u = tf.sigmoid(linear(x, output_size, False))
            with tf.variable_scope("candidate"):
                x = inputs + [r * state]
                c = tf.tanh(linear(x, output_size, True))

            new_state = u * state + (1 - u) * c

        return new_state, new_state

    @property
    def state_size(self):
        return self.size

    @property
    def output_size(self):
        return self.size


def encoder(cell, inputs, sequence_length, hidden_size, dtype=None,
            scope=None):
    dtype = None or inputs.dtype

    with tf.variable_scope(scope or "encoder"):
        outputs = tf.nn.bidirectional_dynamic_rnn(cell, cell,
                                                  inputs,
                                                  sequence_length,
                                                  time_major=True,
                                                  swap_memory=True,
                                                  dtype=dtype)
    return outputs


# precompute mapped attention states to speed up decoding
def map_attention_states(attention_states, attn_size, scope=None):
    with tf.variable_scope(scope or "attention"):
        hidden_size = attention_states.get_shape().as_list()[2]
        shape = tf.shape(attention_states)
        batched_states = tf.reshape(attention_states, [-1, hidden_size])
        mapped_states = linear(batched_states, attn_size, False,
                               scope="annotation_w")
        mapped_states = tf.reshape(mapped_states,
                                   [shape[0], shape[1], attn_size])

    return mapped_states


def attention(query, mapped_states, attn_size, attention_mask=None,
              scope=None):
    with tf.variable_scope(scope or "attention"):
        mapped_query = linear(query, attn_size, False, scope="query_w")
        mapped_query = mapped_query[None, :, :]

        batch = tf.shape(query)[0]
        hidden = tf.tanh(mapped_query + mapped_states)
        hidden = tf.reshape(hidden, [-1, attn_size])

        with tf.variable_scope("attention"):
            score = linear(hidden, 1, False, scope="attention_v")

        exp_score = tf.exp(score)
        exp_score = tf.reshape(exp_score, [-1, batch])

        if attention_mask is not None:
            exp_score = exp_score * attention_mask
            alpha = exp_score / tf.reduce_sum(exp_score, 0)[None, :]

        return alpha[:, :, None]


def decoder(cell, inputs, initial_state, attention_states, attention_length,
            sequence_length, attention_size=None, dtype=None, scope=None):
    if inputs is None:
        raise ValueError("inputs must not be None")

    output_size = cell.output_size
    dtype = dtype or inputs.dtype

    if attention_length is None:
        attention_mask = None
    else:
        attention_mask = tf.sequence_mask(attention_length, dtype=dtype)
        attention_mask = tf.transpose(attention_mask)

    time_steps = tf.shape(inputs)[0]
    batch = tf.shape(inputs)[1]

    if attention_size is not None:
        attn_size = attention_size
    else:
        attn_size = output_size

    zero_output = tf.zeros([batch, output_size], dtype)

    if initial_state is None:
        initial_state = cell.zero_state(batch, dtype)

    with tf.variable_scope(scope or "decoder"):
        mapped_states = map_attention_states(attention_states, attn_size)

        input_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="input_array")
        alpha_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="alpha_array")
        output_ta = tf.TensorArray(tf.float32, time_steps,
                                   tensor_array_name="output_array")
        context_ta = tf.TensorArray(tf.float32, time_steps,
                                    tensor_array_name="context_array")
        input_ta = input_ta.unpack(inputs)

        def loop(time, alpha_ta, output_ta, context_ta, state):
            alpha = attention(state, mapped_states, attn_size, attention_mask)
            context = tf.reduce_sum(alpha * attention_states, 0)

            inputs = input_ta.read(time)
            call_cell = lambda: cell([inputs, context], state)
            output, new_state = rnn_step(time, sequence_length, None, None,
                                         zero_output, state, call_cell, None,
                                         True)
            alpha_ta = alpha_ta.write(time, alpha[:, :, 0])
            output_ta = output_ta.write(time, output)
            context_ta = context_ta.write(time, context)
            return (time + 1, alpha_ta, output_ta, context_ta, new_state)

        time = tf.constant(0, dtype=tf.int32, name="time")
        cond = lambda time, *_: time < time_steps
        loop_vars = (time, alpha_ta, output_ta, context_ta, initial_state)

        outputs = tf.while_loop(cond, loop, loop_vars, parallel_iterations=32,
                                swap_memory=True)

        alpha_final_ta = outputs[1]
        output_final_ta = outputs[2]
        context_final_ta = outputs[3]
        final_state = outputs[4]

        all_alpha = alpha_final_ta.pack()
        all_output = output_final_ta.pack()
        all_context = context_final_ta.pack()

        all_output.set_shape([None, None, output_size])

        return (all_output, all_context, all_alpha, final_state)


class rnnsearch:

    def __init__(self, emb_size, hidden_size, attn_size,
                 svocab_size, tvocab_size, scope=None):
        # training graph
        with tf.variable_scope(scope or "rnnsearch"):
            src_seq = tf.placeholder(tf.int32, [None, None], "soruce_sequence")
            src_len = tf.placeholder(tf.int32, [None], "source_length")
            tgt_seq = tf.placeholder(tf.int32, [None, None], "target_sequence")
            tgt_len = tf.placeholder(tf.int32, [None], "target_length")

            with tf.device("/cpu:0"):
                source_embedding = tf.get_variable("source_embedding",
                                                   [svocab_size, emb_size],
                                                   tf.float32)
                target_embedding = tf.get_variable("target_embedding",
                                                   [tvocab_size, emb_size],
                                                   tf.float32)
                source_bias = tf.get_variable("source_embedding_bias",
                                              [emb_size], tf.float32)
                target_bias = tf.get_variable("target_embedding_bias",
                                              [emb_size], tf.float32)

            source_inputs = tf.gather(source_embedding, src_seq) + source_bias
            target_inputs = tf.gather(target_embedding, tgt_seq) + target_bias

            # run encoder
            cell = gru_cell(hidden_size)
            outputs = encoder(cell, source_inputs, src_len, hidden_size)
            encoder_outputs, encoder_output_states = outputs

            # compute initial state for decoder
            annotation = tf.concat(2, encoder_outputs)
            final_state = encoder_output_states[-1]
            initial_state = tf.tanh(linear(final_state, hidden_size, True,
                                           scope="initial"))

            # run decoder
            decoder_outputs = decoder(cell, target_inputs, initial_state,
                                      annotation, src_len, tgt_len, attn_size)
            all_output, all_context, all_alpha, final_state = decoder_outputs

            # compute costs
            batch = tf.shape(tgt_seq)[1]
            zero_embedding = tf.zeros([1, batch, emb_size])
            shift_inputs = tf.concat(0, [zero_embedding, target_inputs])
            shift_inputs = shift_inputs[:-1, :, :]

            all_states = tf.concat(0, [tf.expand_dims(initial_state, 0),
                                       all_output])
            prev_states = all_states[:-1]

            shift_inputs = tf.reshape(shift_inputs, [-1, emb_size])
            prev_states = tf.reshape(prev_states, [-1, hidden_size])
            all_context = tf.reshape(all_context, [-1, 2 * hidden_size])

            features = [prev_states, shift_inputs, all_context]
            hidden = maxout(features, hidden_size / 2, 2, True)
            readout = linear(hidden, emb_size, False,  scope="deepout")
            logits = linear(readout, tvocab_size, True, scope="prediction")

            labels = tf.reshape(tgt_seq, [-1])
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
                                                                      labels)
            crossent = tf.reshape(crossent, tf.shape(tgt_seq))
            mask = tf.sequence_mask(tgt_len, dtype=tf.float32)

            mask = tf.transpose(mask)
            cost = tf.reduce_mean(tf.reduce_sum(crossent * mask, 0))

        training_inputs = [src_seq, src_len, tgt_seq, tgt_len]
        training_outputs = [cost]
        evaluate = function(training_inputs, training_outputs)

        # encoding
        encoding_inputs = [src_seq, src_len]
        encoding_outputs = [annotation, initial_state]
        encode = function(encoding_inputs, encoding_outputs)

        # decoding graph
        with tf.variable_scope(scope or "rnnsearch", reuse=True):
            prev_words = tf.placeholder(tf.int32, [None], "prev_token")

            with tf.device("/cpu:0"):
                target_embedding = tf.get_variable("target_embedding",
                                                   [tvocab_size, emb_size],
                                                   tf.float32)
                target_bias = tf.get_variable("target_embedding_bias",
                                              [emb_size], tf.float32)

            target_inputs = tf.gather(target_embedding, prev_words)
            target_inputs = target_inputs + target_bias

            # zeros out embedding if y is 0
            cond = tf.equal(prev_words, 0)
            cond = tf.cast(cond, tf.float32)
            target_inputs = target_inputs * (1.0 - tf.expand_dims(cond, 1))

            attention_mask = tf.sequence_mask(src_len, dtype=tf.float32)
            attention_mask = tf.transpose(attention_mask)

            with tf.variable_scope("decoder"):
                mapped_states = map_attention_states(annotation, attn_size)
                alpha = attention(initial_state, mapped_states, attn_size,
                                  attention_mask)
                context = tf.reduce_sum(alpha * annotation, 0)
                output, next_state = cell([target_inputs, context],
                                          initial_state)

            features = [initial_state, target_inputs, context]
            hidden = maxout(features, hidden_size / 2, 2, True)
            readout = linear(hidden, emb_size, False,  scope="deepout")
            logits = linear(readout, tvocab_size, True, scope="prediction")
            probs = tf.nn.softmax(logits)

        precomputation_inputs = [annotation]
        precomputation_outputs = [mapped_states]
        precompute = function(precomputation_inputs, precomputation_outputs)

        alignment_inputs = [initial_state, annotation, mapped_states, src_len]
        alignment_outputs = [alpha, context]
        align = function(alignment_inputs, alignment_outputs)

        prediction_inputs = [prev_words, initial_state, context]
        prediction_outputs = [probs]
        predict = function(prediction_inputs, prediction_outputs)

        generation_inputs = [prev_words, initial_state, context]
        generation_outputs = [next_state]
        generate = function(generation_inputs, generation_outputs)

        self.cost = cost
        self.inputs = training_inputs
        self.outputs = training_outputs
        self.align = align
        self.encode = encode
        self.predict = predict
        self.generate = generate
        self.evaluate = evaluate
        self.precompute = precompute
        self.parameter = tf.trainable_variables()


def beamsearch(model, seq, beamsize=10, normalize=False, maxlen=None,
               minlen=None):
    size = beamsize

    vocabulary = model.option["vocabulary"]
    eos_symbol = model.option["eos"]
    vocab = vocabulary[1][1]
    eosid = vocabulary[1][0][eos_symbol]

    time_dim = 0
    batch_dim = 1

    if maxlen == None:
        maxlen = seq.shape[time_dim] * 3

    if minlen == None:
        minlen = seq.shape[time_dim] / 2

    seq_len = np.array([seq.shape[time_dim]])
    annotation, initial_state = model.encode(seq, seq_len)
    mapped_states = model.precompute(annotation)

    initial_beam = beam(size)
    # </s>
    initial_beam.candidate = [[0]]
    initial_beam.score = np.zeros([1], "float32")

    hypo_list = []
    beam_list = [initial_beam]
    cond = lambda x: x[-1] == eosid

    state = initial_state

    for k in range(maxlen):
        # get previous results
        prev_beam = beam_list[-1]
        candidate = prev_beam.candidate
        num = len(candidate)
        last_words = np.array(map(lambda t: t[-1], candidate), "int32")

        # prediction
        batch_seq_len = np.repeat(seq_len, num, 0)
        batch_annot = np.repeat(annotation, num, batch_dim)
        batch_mannot = np.repeat(mapped_states, num, batch_dim)
        alpha, context = model.align(state, batch_annot, batch_mannot,
                                     batch_seq_len)

        prob_dist = model.predict(last_words, state, context)

        # select nbest
        logprobs = np.log(prob_dist)

        if k < minlen:
            logprobs[:, eosid] = -np.inf

        # force to add eos symbol
        if k == maxlen - 1:
            # copy
            eosprob = logprobs[:, eosid].copy()
            logprobs[:, :] = -np.inf
            logprobs[:, eosid] = eosprob

        next_beam = beam(size)
        outputs = next_beam.prune(logprobs, cond, prev_beam)

        # translation complete
        hypo_list.extend(outputs[0])
        batch_indices, word_indices = outputs[1:]
        size -= len(outputs[0])

        if size == 0:
            break

        state = select_nbest(state, batch_indices)
        context = select_nbest(context, batch_indices)

        # generate next state
        candidate = next_beam.candidate
        num = len(candidate)
        current_words = np.array(map(lambda t: t[-1], candidate), "int32")
        state = model.generate(current_words, state, context)
        beam_list.append(next_beam)

    if len(hypo_list) == 0:
        score_list = [0.0]
        hypo_list = [["<s>"]]
    else:
        score_list = [item[1] for item in hypo_list]
        hypo_list = [item[0] for item in hypo_list]

    for i, (trans, score) in enumerate(zip(hypo_list, score_list)):
        # exclude "<s>"
        count = len(trans) - 1
        if count > 0:
            if normalize:
                score_list[i] = score / count
            else:
                score_list[i] = score

    hypo_list = np.array(hypo_list)[np.argsort(score_list)]
    score_list = np.array(sorted(score_list))

    output = []

    for trans, score in zip(hypo_list, score_list):
        trans = map(lambda x: vocab[x], trans)
        output.append((trans[1:-1], score))

    return output
