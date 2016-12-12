# rnnsearch.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy as np
import tensorflow as tf

from nn import config
from nn import multirnn
from nn import dropout_rnn
from nn import scan, function
from nn import  gru, gru_config
from nn import linear, linear_config
from nn import feedforward, feedforward_config
from nn import embedding, embedding_config
from nn import maxout, maxout_config
from utils import get_or_default, add_if_not_exsit


class encoder_config(config):
    """
    * dtype: str, default tf.float32
    * scope: str, default "encoder"
    * forward_rnn: gru_config, config behavior of forward rnn
    * backward_rnn: gru_config, config behavior of backward rnn
    """

    def __init__(self, **kwargs):
        self.dtype = get_or_default(kwargs, "dtype", tf.float32)
        self.scope = get_or_default(kwargs, "scope", "encoder")
        self.layers = 2
        self.keep_prob = 0.5
        self.forward_rnn = gru_config(dtype=self.dtype, scope="forward_rnn")
        self.backward_rnn = gru_config(dtype=self.dtype, scope="backward_rnn")


class decoder_config(config):
    """
    * dtype: str, default tf.float32
    * scope: str, default "decoder"
    * init_transform: feedforward_config, config initial state transform
    * annotation_transform: linear_config, config annotation transform
    * state_transform: linear_config, config state transform
    * context_transform: linear_config, config context transform
    * rnn: gru_config, config decoder rnn
    * maxout: maxout_config, config maxout unit
    * deepout: linear_config, config deepout transform
    * classify: linear_config, config classify transform
    """

    def __init__(self, **kwargs):
        self.dtype = get_or_default(kwargs, "dtype", tf.float32)
        self.scope = get_or_default(kwargs, "scope", "decoder")
        self.layers = 2
        self.keep_prob = 0.5
        self.init_transform = feedforward_config(dtype=self.dtype,
                                                 scope="init_transform",
                                                 activation=tf.tanh)
        self.annotation_transform = linear_config(dtype=self.dtype,
                                                  scope="annotation_transform")
        self.state_transform = linear_config(dtype=self.dtype,
                                             scope="state_transform")
        self.context_transform = linear_config(dtype=self.dtype,
                                               scope="context_transform")
        self.rnn = gru_config(dtype=self.dtype, scope="rnn")
        self.maxout = maxout_config(dtype=self.dtype, scope="maxout")
        self.deepout = linear_config(dtypde=self.dtype, scope="deepout")
        self.classify = linear_config(dtype=self.dtype, scope="classify")


class rnnsearch_config(config):
    """
    * dtype: str, default tf.float32
    * scope: str, default "rnnsearch"
    * source_embedding: embedding_config, config source side embedding
    * target_embedding: embedding_config, config target side embedding
    * encoder: encoder_config, config encoder
    * decoder: decoder_config, config decoder
    """

    def __init__(self, **kwargs):
        self.dtype = get_or_default(kwargs, "dtype", tf.float32)
        self.scope = get_or_default(kwargs, "scope", "rnnsearch")
        self.source_embedding = embedding_config(dtype=self.dtype,
                                                 scope="source_embedding")
        self.target_embedding = embedding_config(dtype=self.dtype,
                                                 scope="target_embedding")
        self.encoder = encoder_config(dtype=self.dtype)
        self.decoder = decoder_config(dtype=self.dtype)


# standard rnnsearch configuration (groundhog version)
def get_config():
    config = rnnsearch_config()

    config["*/concat"] = False
    config["*/output_major"] = False

    # embedding
    config["source_embedding/bias/use"] = True
    config["target_embedding/bias/use"] = True

    # encoder
    config["encoder/forward_rnn/reset_gate/bias/use"] = False
    config["encoder/forward_rnn/update_gate/bias/use"] = False
    config["encoder/forward_rnn/candidate/bias/use"] = True
    config["encoder/backward_rnn/reset_gate/bias/use"] = False
    config["encoder/backward_rnn/update_gate/bias/use"] = False
    config["encoder/backward_rnn/candidate/bias/use"] = True

    # decoder
    config["decoder/init_transform/bias/use"] = True
    config["decoder/annotation_transform/bias/use"] = False
    config["decoder/state_transform/bias/use"] = False
    config["decoder/context_transform/bias/use"] = False
    config["decoder/rnn/reset_gate/bias/use"] = False
    config["decoder/rnn/update_gate/bias/use"] = False
    config["decoder/rnn/candidate/bias/use"] = True
    config["decoder/maxout/bias/use"] = True
    config["decoder/deepout/bias/use"] = False
    config["decoder/classify/bias/use"] = True

    return config


class encoder:

    def __init__(self, input_size, hidden_size, config=encoder_config()):
        scope = config.scope
        num_layers = config.layers

        fwd_cells = []
        bwd_cells = []

        with tf.variable_scope(scope):
            fwd_rnn = gru(input_size, hidden_size, config.forward_rnn)
            bwd_rnn = gru(input_size, hidden_size, config.backward_rnn)
            # add dropout
            fwd_rnn = dropout_rnn(fwd_rnn, config.keep_prob)
            bwd_rnn = dropout_rnn(fwd_rnn, config.keep_prob)

            fwd_cells.append(fwd_rnn)
            bwd_cells.append(bwd_rnn)

            for i in range(1, num_layers):
                fwd_rnn = gru(hidden_size, hidden_size, config.forward_rnn)
                bwd_rnn = gru(hidden_size, hidden_size, config.backward_rnn)
                fwd_rnn = dropout_rnn(fwd_rnn, config.keep_prob)
                bwd_rnn = dropout_rnn(fwd_rnn, config.keep_prob)
                fwd_cells.append(fwd_rnn)
                bwd_cells.append(bwd_rnn)

        forward_rnn = multirnn(fwd_cells)
        backward_rnn = multirnn(bwd_cells)

        params = []
        params.extend(forward_rnn.parameter)
        params.extend(backward_rnn.parameter)

        def control_dropout(enable=True):
            for cell in fwd_cells:
                cell.enable = enable
            for cell in bwd_cells:
                cell.enable = enable

        def forward(x, mask):
            batch = tf.shape(x)[1]

            init_output = tf.zeros([batch, hidden_size])
            fwd_init_state = forward_rnn.make_state(batch)
            bwd_init_state = backward_rnn.make_state(batch)

            def forward_step(emb, mask, output, *state):
                new_output, new_state = forward_rnn(emb, state)
                mask = tf.expand_dims(mask, 1)
                new_output = (1.0 - mask) * output + mask * new_output
                new_state = [(1.0 - mask) * o + mask * n for o, n in
                             zip(state, new_state)]
                return [new_output] + new_state

            def backward_step(emb, mask, output, *state):
                new_output, new_state = backward_rnn(emb, state)
                mask = tf.expand_dims(mask, 1)
                new_output = (1.0 - mask) * output + mask * new_output
                new_state = [(1.0 - mask) * o + mask * n for o, n in
                             zip(state, new_state)]
                return [new_output] + new_state

            seq = [x, mask]
            outputs_info = [init_output] + list(fwd_init_state)
            outputs = scan(forward_step, seq, outputs_info)
            hf = outputs[0]

            # reverse sequence
            seq = [x[::-1], mask[::-1]]
            outputs_info = [init_output] + list(bwd_init_state)
            outputs = scan(backward_step, seq, outputs_info)
            hb = outputs[0]
            hb = hb[::-1]

            return tf.concat(2, [hf, hb])

        self.name = scope
        self.config = config
        self.forward = forward
        self.parameter = params
        self.control_dropout = control_dropout

    def __call__(self, x, mask):
        return self.forward(x, mask)


class decoder:

    def __init__(self, emb_size, shidden_size, thidden_size, ahidden_size,
                 mhidden_size, maxpart, dhidden_size, voc_size,
                 config=decoder_config()):
        scope = config.scope
        num_layers = config.layers
        ctx_size = 2 * shidden_size

        cells = []

        with tf.variable_scope(scope):
            init_transform = feedforward(shidden_size, thidden_size,
                                         config.init_transform)
            annotation_transform = linear(ctx_size, ahidden_size,
                                          config.annotation_transform)
            state_transform = linear(thidden_size, ahidden_size,
                                     config.state_transform)
            context_transform = linear(ahidden_size, 1,
                                       config.context_transform)

            rnn = gru([emb_size, ctx_size], thidden_size, config.rnn)
            rnn = dropout_rnn(rnn, config.keep_prob)
            cells.append(rnn)

            for i in range(1, num_layers):
                rnn = gru([thidden_size, ctx_size], thidden_size, config.rnn)
                rnn = dropout_rnn(rnn, config.keep_prob)
                cells.append(rnn)

            maxout_transform = maxout([thidden_size, emb_size, ctx_size],
                                      mhidden_size, maxpart, config.maxout)
            deepout_transform = linear(mhidden_size, dhidden_size,
                                       config.deepout)
            classify_transform = linear(dhidden_size, voc_size,
                                        config.classify)

        def control_dropout(enable=True):
            for cell in cells:
                cell.enable = enable

        rnn = multirnn(cells)

        params = []
        params.extend(init_transform.parameter)
        params.extend(annotation_transform.parameter)
        params.extend(state_transform.parameter)
        params.extend(context_transform.parameter)
        params.extend(rnn.parameter)
        params.extend(maxout_transform.parameter)
        params.extend(deepout_transform.parameter)
        params.extend(classify_transform.parameter)

        def attention(state, xmask, mapped_annotation):
            mapped_state = state_transform(state)
            hidden = tf.tanh(mapped_state + mapped_annotation)
            score = context_transform(hidden)
            shape = tf.shape(score)
            score = tf.reshape(score, [shape[0], shape[1]])
            # softmax over masked batch
            alpha = tf.exp(score)
            alpha = alpha * xmask
            alpha = alpha / tf.reduce_sum(alpha, 0)
            return alpha

        def compute_initstate(annotation):
            hf, hb = tf.split(2, 2, annotation)
            inis = init_transform(hb[0])
            mapped_annotation = annotation_transform(annotation)

            return inis, mapped_annotation

        def compute_context(state, xmask, annotation, mapped_annotation):
            alpha = attention(state, xmask, mapped_annotation)
            alpha = tf.expand_dims(alpha, 2)
            context = tf.reduce_sum(alpha * annotation, 0)
            return [alpha, context]

        def compute_probability(yemb, state, context):
            maxhid = maxout_transform([state, yemb, context])
            readout = deepout_transform(maxhid)
            logit = classify_transform(readout)
            prob = tf.nn.softmax(logit)

            return prob

        def compute_state(yemb, ymask, context, output, *state):
            # [input_below, [[other_inputs_1], [other_inputs_2] ...]]
            inputs = [yemb, [[context] for i in range(num_layers)]]
            new_output, new_state = rnn(inputs, state)
            ymask = tf.expand_dims(ymask, 1)
            new_output = (1.0 - ymask) * output + ymask * new_output
            new_state = [(1.0 - ymask) * o + ymask * n for o, n in
                         zip(state, new_state)]

            return [new_output] + list(new_state)

        def compute_attention_score(yseq, xmask, ymask, annotation):
            batch = tf.shape(yseq)[1]
            init_state = rnn.make_state(batch)
            init_output, mapped_annotation = compute_initstate(annotation)

            def step(yemb, ymask, output, *state_and_nonseq):
                state = state_and_nonseq[:-3]
                xmask, annotation, mannotation = state_and_nonseq[-3:]
                outs = compute_context(output, xmask, annotation, mannotation)
                alpha, context = outs
                outs = compute_state(yemb, ymask, context, output, *state)
                new_output = outs[0]
                new_state = outs[1:]
                return [alpha, new_output] + list(new_state)

            seq = [yseq, ymask]
            oinfo = [None, init_output] + list(init_state)
            nonseq = [xmask, annotation, mapped_annotation]
            outputs = scan(step, seq, oinfo, nonseq)

            return outputs[0]

        def forward(yseq, xmask, ymask, annotation):
            batch = tf.shape(yseq)[1]

            init_emb = tf.zeros([1, batch, emb_size])
            yshift = tf.concat(0, [init_emb, yseq])
            yshift = yshift[:-1]

            init_state = rnn.make_state(batch)
            init_output, mapped_annotation = compute_initstate(annotation)

            def step(yemb, ymask, output, *state_and_nonseq):
                state = state_and_nonseq[:-3]
                xmask, annotation, mannotation = state_and_nonseq[-3:]
                outs = compute_context(output, xmask, annotation, mannotation)
                alpha, context = outs
                outs = compute_state(yemb, ymask, context, output, *state)
                new_output = outs[0]
                new_state = outs[1:]
                return [context, new_output] + list(new_state)

            seq = [yseq, ymask]
            oinfo = [None, init_output] + list(init_state)
            nonseq = [xmask, annotation, mapped_annotation]
            outputs = scan(step, seq, oinfo, nonseq)
            contexts = outputs[0]
            states = outputs[1]

            inis = tf.expand_dims(init_output, 0)
            all_states = tf.concat(0, [inis, states])
            prev_states = all_states[:-1]

            maxhid = maxout_transform([prev_states, yshift, contexts])
            readout = deepout_transform(maxhid)
            logit = classify_transform(readout)
            shape = tf.shape(logit)
            logit = tf.reshape(logit, [shape[0] * shape[1], -1])

            return logit

        self.name = scope
        self.config = config
        self.forward = forward
        self.parameter = params
        self.state_size = rnn.state_size
        self.make_state = rnn.make_state
        self.control_dropout = control_dropout
        self.compute_initstate = compute_initstate
        self.compute_context = compute_context
        self.compute_probability = compute_probability
        self.compute_state = compute_state
        self.compute_attention_score = compute_attention_score

    def __call__(self, yseq, xmask, ymask, annotation):
        return self.forward(yseq, xmask, ymask, annotation)


class rnnsearch:

    def __init__(self, config=get_config(), **option):
        scope = config.scope

        sedim, tedim = option["embdim"]
        shdim, thdim, ahdim = option["hidden"]
        maxdim = option["maxhid"]
        deephid = option["deephid"]
        k = option["maxpart"]
        svocab, tvocab = option["vocabulary"]
        sw2id, sid2w = svocab
        tw2id, tid2w = tvocab
        svsize = len(sid2w)
        tvsize = len(tid2w)

        with tf.variable_scope(scope):
            source_embedding = embedding(svsize, sedim,
                                         config.source_embedding)
            target_embedding = embedding(tvsize, tedim,
                                         config.target_embedding)
            rnn_encoder = encoder(sedim, shdim, config.encoder)
            rnn_decoder = decoder(tedim, shdim, thdim, ahdim, maxdim, k,
                                  deephid, tvsize, config.decoder)

        params = []
        params.extend(source_embedding.parameter)
        params.extend(target_embedding.parameter)
        params.extend(rnn_encoder.parameter)
        params.extend(rnn_decoder.parameter)

        def training_graph():
            xseq = tf.placeholder(tf.int32, [None, None])
            xmask = tf.placeholder(tf.float32, [None, None])
            yseq = tf.placeholder(tf.int32, [None, None])
            ymask = tf.placeholder(tf.float32, [None, None])

            rnn_encoder.control_dropout(enable=True)
            rnn_decoder.control_dropout(enable=True)

            xemb = source_embedding(xseq)
            yemb = target_embedding(yseq)

            annotation = rnn_encoder(xemb, xmask)
            logits = rnn_decoder(yemb, xmask, ymask, annotation)

            labels = tf.reshape(yseq, [-1])
            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
                                                                  labels)
            cost = tf.reshape(cost, tf.shape(yseq))
            cost = tf.reduce_sum(cost * ymask, 0)
            cost = tf.reduce_mean(cost)

            return [xseq, xmask, yseq, ymask], [cost]

        def evaluation_graph():
            xseq = tf.placeholder(tf.int32, [None, None])
            xmask = tf.placeholder(tf.float32, [None, None])
            yseq = tf.placeholder(tf.int32, [None, None])
            ymask = tf.placeholder(tf.float32, [None, None])

            rnn_encoder.control_dropout(enable=False)
            rnn_decoder.control_dropout(enable=False)

            xemb = source_embedding(xseq)
            yemb = target_embedding(yseq)

            annotation = rnn_encoder(xemb, xmask)
            logits = rnn_decoder(yemb, xmask, ymask, annotation)

            labels = tf.reshape(yseq, [-1])
            logp = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
                                                                  labels)
            logp = tf.reshape(logp, tf.shape(yseq))
            logp = tf.reduce_sum(logp * ymask, 0)

            return [xseq, xmask, yseq, ymask], [logp]

        def mrt_training_graph():
            xseq = tf.placeholder(tf.int32, [None])
            yseq = tf.placeholder(tf.int32, [None, None])
            ymask = tf.placeholder(tf.float32, [None, None])
            loss = tf.placeholder(tf.float32, [None])
            sharp = tf.placeholder(tf.float32, [])

            batch = tf.shape(yseq)[1]

            rnn_encoder.control_dropout(enable=True)
            rnn_decoder.control_dropout(enable=True)

            # expand dim
            xseqs = xseq[:, None]
            xmask = tf.ones([tf.shape(xseq)[0], 1], dtype=tf.float32)

            xemb = source_embedding(xseqs)

            annotation = rnn_encoder(xemb, xmask)

            annotation = tf.tile(annotation, [1, batch, 1])
            xmask = tf.tile(xmask, [1, batch])

            yemb = target_embedding(yseq)
            logits = rnn_decoder(yemb, xmask, ymask, annotation)

            # calculate mrt cost
            labels = tf.reshape(yseq, [-1])
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
                                                                      labels)
            log_probs = tf.reshape(-crossent, tf.shape(yseq))
            # log_probs => [batch,]
            log_probs = tf.reduce_sum(log_probs * ymask, 0)

            score = log_probs * sharp
            # safe softmax
            score = score - tf.reduce_min(score)
            score = tf.exp(score)
            qprob = score / tf.reduce_sum(score)
            risk = tf.reduce_sum(qprob * loss)

            return [xseq, yseq, ymask, loss, sharp], [risk]

        def encode_graph():
            xseq = tf.placeholder(tf.int32, [None, None])
            xmask = tf.placeholder(tf.float32, [None, None])

            rnn_encoder.control_dropout(enable=False)

            xemb = source_embedding(xseq)
            annotation = rnn_encoder(xemb, xmask)

            return [xseq, xmask], [annotation]

        def initstate_graph():
            annotation = tf.placeholder(tf.float32, [None, None, 2 * shdim])

            rnn_decoder.control_dropout(enable=False)

            # initstate, mapped_annotation
            init_output, mannot = rnn_decoder.compute_initstate(annotation)
            init_state = rnn_decoder.make_state(tf.shape(annotation)[1])

            output = [mannot, init_output] + list(init_state)

            return [annotation], output

        def context_graph():
            state = tf.placeholder(tf.float32, [None, thdim])
            xmask = tf.placeholder(tf.float32, [None, None])
            annotation = tf.placeholder(tf.float32, [None, None, 2 * shdim])
            mannotation = tf.placeholder(tf.float32, [None, None, ahdim])

            rnn_decoder.control_dropout(enable=False)

            inputs = [state, xmask, annotation, mannotation]
            alpha, context = rnn_decoder.compute_context(*inputs)

            return inputs, [alpha, context]

        def probability_graph():
            y = tf.placeholder(tf.int32, [None])
            state = tf.placeholder(tf.float32, [None, thdim])
            context = tf.placeholder(tf.float32, [None, 2 * shdim])

            rnn_decoder.control_dropout(enable=False)

            # 0 for initial index
            cond = tf.equal(y, 0)
            yemb = target_embedding(y)
            # zeros out embedding if y is 0
            cond = tf.cast(cond, tf.float32)
            yemb = yemb * (1.0 - tf.expand_dims(cond, 1))
            probs = rnn_decoder.compute_probability(yemb, state, context)

            return [y, state, context], [probs]

        def state_graph():
            y = tf.placeholder(tf.int32, [None])
            ymask = tf.placeholder(tf.float32, [None])
            context = tf.placeholder(tf.float32, [None, 2 * shdim])
            output = tf.placeholder(tf.float32, [None, thdim])
            state_size = rnn_decoder.state_size
            states = [tf.placeholder(tf.float32, [None, size])
                      for size in state_size]

            rnn_decoder.control_dropout(enable=False)

            yemb = target_embedding(y)
            inputs = [yemb, ymask, context, output] + states
            outputs = rnn_decoder.compute_state(*inputs)

            inputs = [y, ymask, context, output] + states

            return inputs, outputs

        def attention_graph():
            xseq = tf.placeholder(tf.int32, [None, None])
            xmask = tf.placeholder(tf.float32, [None, None])
            yseq = tf.placeholder(tf.int32, [None, None])
            ymask = tf.placeholder(tf.float32, [None, None])

            rnn_encoder.control_dropout(enable=False)
            rnn_decoder.control_dropout(enable=False)

            xemb = source_embedding(xseq)
            yemb = target_embedding(yseq)

            annotation = rnn_encoder(xemb, xmask)
            alpha = rnn_decoder.compute_attention_score(yemb, xmask, ymask,
                                                        annotation)

            return function([xseq, xmask, yseq, ymask], alpha)


        def sampling_graph():
            xseq = tf.placeholder(tf.int32, [None, None])
            xmask = tf.placeholder(tf.float32, [None, None])
            maxlen = tf.placeholder(tf.int32, [])

            rnn_encoder.control_dropout(enable=False)
            rnn_decoder.control_dropout(enable=False)

            # expand dim
            batch = tf.shape(xseq)[1]
            xemb = source_embedding(xseq)

            annot = rnn_encoder(xemb, xmask)

            ymask = tf.ones([batch], dtype=tf.float32)
            init_output, mannot = rnn_decoder.compute_initstate(annot)
            init_state = rnn_decoder.make_state(batch)

            def sample_step(pemb, output, *states_and_nonseq):
                state = states_and_nonseq[:-4]
                xmask, ymask, annot, mannot = states_and_nonseq[-4:]
                alpha, context = rnn_decoder.compute_context(output, xmask,
                                                             annot, mannot)
                probs = rnn_decoder.compute_probability(pemb, output, context)
                logprobs = tf.log(probs)
                next_words = tf.multinomial(logprobs, 1)
                next_words = tf.squeeze(next_words, [1])
                yemb = target_embedding(next_words)
                outputs = rnn_decoder.compute_state(yemb, ymask, context,
                                                    output, *state)
                new_output = outputs[0]
                new_state = outputs[1:]
                next_words = tf.cast(next_words, tf.float32)
                return [next_words, yemb, new_output] + list(new_state)

            iemb = tf.zeros([batch, tedim], tf.float32)

            seqs = []
            outputs_info = [None, iemb, init_output] + list(init_state)
            nonseqs = [xmask, ymask, annot, mannot]

            outputs = scan(sample_step, seqs, outputs_info,
                           nonseqs, n_steps=maxlen)

            words = tf.cast(outputs[0], tf.int32)

            return function([xseq, xmask, maxlen], words)

        train_inputs, train_outputs = training_graph()
        mrt_inputs, mrt_outputs = mrt_training_graph()

        def build_function(graph_fn):
            inputs, outputs = graph_fn()
            return function(inputs, outputs)

        evaluate = build_function(evaluation_graph)

        functions = []
        functions.append(build_function(encode_graph))
        functions.append(build_function(initstate_graph))
        functions.append(build_function(context_graph))
        functions.append(build_function(probability_graph))
        functions.append(build_function(state_graph))

        def switch(mrt=False):
            if mrt:
                self.inputs = mrt_inputs
                self.outputs = mrt_outputs
                self.cost = mrt_outputs[0]
            else:
                self.cost = train_outputs[0]
                self.inputs = train_inputs
                self.outputs = train_outputs

        self.name = scope
        self.config = config
        self.parameter = params
        self.option = option
        self.cost = train_outputs[0]
        self.inputs = train_inputs
        self.outputs = train_outputs
        self.updates = []
        self.search = functions
        self.evaluate = evaluate
        self.sampler = sampling_graph()
        self.attention = attention_graph()
        self.switch = switch
        self.encoder = rnn_encoder
        self.decoder = rnn_decoder
        self.embedding = [source_embedding, target_embedding]


# based on groundhog's impelmentation
def beamsearch(models, xseq, **option):
    add_if_not_exsit(option, "beamsize", 10)
    add_if_not_exsit(option, "normalize", False)
    add_if_not_exsit(option, "maxlen", None)
    add_if_not_exsit(option, "minlen", None)
    add_if_not_exsit(option, "arithmetric", False)

    if not isinstance(models, (list, tuple)):
        models = [models]

    num_models = len(models)
    functions = [model.search for model in models]

    vocabulary = models[0].option["vocabulary"]
    eos = models[0].option["eos"]
    vocab = vocabulary[1][1]
    eosid = vocabulary[1][0][eos]
    state_size = models[0].decoder.state_size

    size = option["beamsize"]
    maxlen = option["maxlen"]
    minlen = option["minlen"]
    normalize = option["normalize"]
    arithmetric = option["arithmetric"]

    if maxlen == None:
        maxlen = len(xseq) * 3

    if minlen == None:
        minlen = len(xseq) / 2

    annot = [None for i in range(num_models)]
    mannot = [None for i in range(num_models)]
    contexts = [None for i in range(num_models)]
    outputs = [None for i in range(num_models)]
    states = [None for i in range(num_models)]
    probs = [None for i in range(num_models)]

    xmask = np.ones(xseq.shape, "float32")

    for i in range(num_models):
        encode = functions[i][0]
        compute_istate = functions[i][1]
        annot[i] = encode(xseq, xmask)
        outs = compute_istate(annot[i])
        mannot[i] = outs[0]
        outputs[i] = outs[1]
        states[i] = outs[1:]

    hdim = outputs[0].shape[1]
    cdim = annot[0].shape[2]

    trans = [[]]
    costs = [0.0]
    final_trans = []
    final_costs = []

    for k in range(maxlen):
        if size == 0:
            break

        # current translation number
        num = len(trans)

        if k > 0:
            last_words = np.array(map(lambda t: t[-1], trans))
            last_words = last_words.astype("int32")
        else:
            last_words = np.zeros(num, "int32")

        xmasks = np.repeat(xmask, num, 1)
        ymask = np.ones((num,), "float32")
        annots = [np.repeat(annot[i], num, 1) for i in range(num_models)]
        mannots = [np.repeat(mannot[i], num, 1) for i in range(num_models)]

        for i in range(num_models):
            compute_context = functions[i][2]
            alpha, contexts[i] = compute_context(outputs[i], xmasks, annots[i],
                                                 mannots[i])

        for i in range(num_models):
            compute_probs = functions[i][3]
            probs[i] = compute_probs(last_words, outputs[i], contexts[i])

        if arithmetric:
            logprobs = np.log(sum(probs) / num_models)
        else:
            # geometric mean
            logprobs = sum(np.log(probs)) / num_models

        if k < minlen:
            logprobs[:, eosid] = -np.inf

        # force to add eos symbol
        if k == maxlen - 1:
            # copy
            eosprob = logprobs[:, eosid].copy()
            logprobs[:, :] = -np.inf
            logprobs[:, eosid] = eosprob

        ncosts = np.array(costs)[:, None] - logprobs
        fcosts = ncosts.flatten()
        nbest = np.argpartition(fcosts, size)[:size]

        vocsize = logprobs.shape[1]
        tinds = nbest / vocsize
        winds = nbest % vocsize
        costs = fcosts[nbest]

        newtrans = [[]] * size
        newcosts = np.zeros(size)
        newoutputs = np.zeros((num_models, size, hdim), "float32")
        newcontexts = np.zeros((num_models, size, cdim), "float32")

        new_states = [[] for i in range(num_models)]

        for dim in state_size:
            array = np.zeros((size, dim), "float32")
            for i in range(num_models):
                new_states[i].append(array)

        inputs = np.zeros(size, "int32")

        for i, (idx, nword, ncost) in enumerate(zip(tinds, winds, costs)):
            newtrans[i] = trans[idx] + [nword]
            newcosts[i] = ncost
            for j in range(num_models):
                newoutputs[j][i] = outputs[j][idx]
                newcontexts[j][i] = contexts[j][idx]
                for k in range(len(state_size)):
                    new_states[j][k][i] = states[j][k][idx]

            inputs[i] = nword

        ymask = np.ones((size,), "float32")

        next_output = []
        next_states = []

        for i in range(num_models):
            compute_state = functions[i][-1]
            outs = compute_state(inputs, ymask, newcontexts[i], newoutputs[i],
                                 *new_states[i])
            next_output.append(outs[0])
            next_states.append(outs[1:])

        trans = []
        costs = []
        indices = []

        for i in range(size):
            if newtrans[i][-1] != eosid:
                trans.append(newtrans[i])
                costs.append(newcosts[i])
                indices.append(i)
            else:
                size -= 1
                final_trans.append(newtrans[i])
                final_costs.append(newcosts[i])

        outputs = [next_output[i][indices] for i in range(num_models)]
        states = [[] for i in range(num_models)]

        for i in range(num_models):
            for j in range(len(state_size)):
                states[i].append(next_states[i][j][indices])

    if len(final_trans) == 0:
        final_trans = [[]]
        final_costs = [0.0]

    for i, (cost, trans) in enumerate(zip(final_costs, final_trans)):
        count = len(trans)
        if count > 0:
            if normalize:
                final_costs[i] = cost / count
            else:
                final_costs[i] = cost

    final_trans = np.array(final_trans)[np.argsort(final_costs)]
    final_costs = np.array(sorted(final_costs))

    translations = []

    for cost, trans in zip(final_costs, final_trans):
        trans = map(lambda x: vocab[x], trans)
        translations.append((trans, cost))

    return translations


def batchsample(model, xseq, xmask, **option):
    add_if_not_exsit(option, "maxlen", None)
    maxlen = option["maxlen"]

    sampler = model.sampler

    vocabulary = model.option["vocabulary"]
    eos = model.option["eos"]
    vocab = vocabulary[1][1]
    eosid = vocabulary[1][0][eos]

    if maxlen == None:
        maxlen = int(len(xseq) * 1.5)

    words = sampler(xseq, xmask, maxlen)
    trans = words.astype("int32")

    samples = []

    for i in range(trans.shape[1]):
        example = trans[:, i]
        # remove <eos> symbol
        index = -1

        for i in range(len(example)):
            if example[i] == eosid:
                index = i
                break

        if index > 0:
            example = example[:index]

        example = map(lambda x: vocab[x], example)

        samples.append(example)

    return samples
