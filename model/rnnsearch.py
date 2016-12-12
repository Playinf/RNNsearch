# rnnsearch.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy as np
import tensorflow as tf

from nn import config
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

        with tf.variable_scope(scope):
            forward_encoder = gru(input_size, hidden_size, config.forward_rnn)
            backward_encoder = gru(input_size, hidden_size,
                                   config.backward_rnn)

        params = []
        params.extend(forward_encoder.parameter)
        params.extend(backward_encoder.parameter)

        def forward(x, mask, initstate):
            def forward_step(x, m, h):
                nh, states = forward_encoder(x, h)
                m = tf.expand_dims(m, 1)
                nh = (1.0 - m) * h + m * nh
                return [nh]

            def backward_step(x, m, h):
                nh, states = backward_encoder(x, h)
                m = tf.expand_dims(m, 1)
                nh = (1.0 - m) * h + m * nh
                return [nh]

            seq = [x, mask]
            hf = scan(forward_step, seq, [initstate])

            # reverse sequence
            seq = [x[::-1], mask[::-1]]
            hb = scan(backward_step, seq, [initstate])
            hb = hb[::-1]

            return tf.concat(2, [hf, hb])

        self.name = scope
        self.config = config
        self.forward = forward
        self.parameter = params

    def __call__(self, x, mask, initstate):
        return self.forward(x, mask, initstate)


class decoder:

    def __init__(self, emb_size, shidden_size, thidden_size, ahidden_size,
                 mhidden_size, maxpart, dhidden_size, voc_size,
                 config=decoder_config()):
        scope = config.scope

        ctx_size = 2 * shidden_size

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
            maxout_transform = maxout([thidden_size, emb_size, ctx_size],
                                      mhidden_size, maxpart, config.maxout)
            deepout_transform = linear(mhidden_size, dhidden_size,
                                       config.deepout)
            classify_transform = linear(dhidden_size, voc_size,
                                        config.classify)

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

        def compute_state(yemb, ymask, state, context):
            new_state, states = rnn([yemb, context], state)
            ymask = tf.expand_dims(ymask, 1)
            new_state = (1.0 - ymask) * state + ymask * new_state

            return new_state

        def compute_attention_score(yseq, xmask, ymask, annotation):
            initstate, mapped_annotation = compute_initstate(annotation)

            def step(yemb, ymask, state, xmask, annotation, mannotation):
                outs = compute_context(state, xmask, annotation, mannotation)
                alpha, context = outs
                new_state = compute_state(yemb, ymask, state, context)
                return [new_state, alpha]

            seq = [yseq, ymask]
            oinfo = [initstate, None]
            nonseq = [xmask, annotation, mapped_annotation]
            states, alpha = scan(step, seq, oinfo, nonseq)

            return alpha

        def forward(yseq, xmask, ymask, annotation):
            batch = tf.shape(yseq)[1]

            init_emb = tf.zeros([1, batch, emb_size])
            yshift = tf.concat(0, [init_emb, yseq])
            yshift = yshift[:-1]

            initstate, mapped_annotation = compute_initstate(annotation)

            def step(yemb, ymask, state, xmask, annotation, mannotation):
                outs = compute_context(state, xmask, annotation, mannotation)
                alpha, context = outs
                new_state = compute_state(yemb, ymask, state, context)
                return [new_state, context]

            seq = [yseq, ymask]
            oinfo = [initstate, None]
            nonseq = [xmask, annotation, mapped_annotation]
            states, contexts = scan(step, seq, oinfo, nonseq)

            inis = tf.expand_dims(initstate, 0)
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

        def mle_training_graph():
            xseq = tf.placeholder(tf.int32, [None, None])
            xmask = tf.placeholder(tf.float32, [None, None])
            yseq = tf.placeholder(tf.int32, [None, None])
            ymask = tf.placeholder(tf.float32, [None, None])

            xemb = source_embedding(xseq)
            yemb = target_embedding(yseq)
            initstate = tf.zeros([tf.shape(xemb)[1], shdim])

            annotation = rnn_encoder(xemb, xmask, initstate)
            logits = rnn_decoder(yemb, xmask, ymask, annotation)

            labels = tf.reshape(yseq, [-1])
            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
                                                                  labels)
            cost = tf.reshape(cost, tf.shape(yseq))
            cost = tf.reduce_sum(cost * ymask, 0)
            cost = tf.reduce_mean(cost)

            return [xseq, xmask, yseq, ymask], [cost]

        def mrt_training_graph():
            xseq = tf.placeholder(tf.int32, [None])
            yseq = tf.placeholder(tf.int32, [None, None])
            ymask = tf.placeholder(tf.float32, [None, None])
            loss = tf.placeholder(tf.float32, [None])
            sharp = tf.placeholder(tf.float32, [])

            batch = tf.shape(yseq)[1]

            # expand dim
            xseqs = xseq[:, None]
            xmask = tf.ones([tf.shape(xseq)[0], 1], dtype=tf.float32)

            xemb = source_embedding(xseqs)
            initstate = tf.zeros([1, shdim])

            annotation = rnn_encoder(xemb, xmask, initstate)

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

        def encoding_graph():
            xseq = tf.placeholder(tf.int32, [None, None])
            xmask = tf.placeholder(tf.float32, [None, None])

            xemb = source_embedding(xseq)
            initstate = tf.zeros([tf.shape(xseq)[1], shdim])
            annotation = rnn_encoder(xemb, xmask, initstate)

            return [xseq, xmask], [annotation]

        def initstate_graph():
            annotation = tf.placeholder(tf.float32, [None, None, None])

            # initstate, mapped_annotation
            outputs = rnn_decoder.compute_initstate(annotation)

            return [annotation], [outputs]

        def context_graph():
            state = tf.placeholder(tf.float32, [None, None])
            xmask = tf.placeholder(tf.float32, [None, None])
            annotation = tf.placeholder(tf.float32, [None, None, None])
            mannotation = tf.placeholder(tf.float32, [None, None, None])

            inputs = [state, xmask, annotation, mannotation]
            alpha, context = rnn_decoder.compute_context(*inputs)

            return inputs, [alpha, context]

        def prediction_graph():
            y = tf.placeholder(tf.int32, [None])
            state = tf.placeholder(tf.float32, [None, None])
            context = tf.placeholder(tf.float32, [None, None])

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
            state = tf.placeholder(tf.float32, [None, None])
            context = tf.placeholder(tf.float32, [None, None])

            yemb = target_embedding(y)
            inputs = [yemb, ymask, state, context]
            new_state = rnn_decoder.compute_state(*inputs)

            return [y, ymask, state, context], [new_state]

        def attention_graph():
            xseq = tf.placeholder(tf.int32, [None, None])
            xmask = tf.placeholder(tf.float32, [None, None])
            yseq = tf.placeholder(tf.int32, [None, None])
            ymask = tf.placeholder(tf.float32, [None, None])

            xemb = source_embedding(xseq)
            yemb = target_embedding(yseq)
            initstate = tf.zeros([tf.shape(xemb)[1], shdim])

            annotation = rnn_encoder(xemb, xmask, initstate)
            alpha = rnn_decoder.compute_attention_score(yemb, xmask, ymask,
                                                        annotation)

            return [xseq, xmask, yseq, ymask], [alpha]

        def sampling_graph():
            xseq = tf.placeholder(tf.int32, [None, None])
            xmask = tf.placeholder(tf.float32, [None, None])
            maxlen = tf.placeholder(tf.int32, [])

            batch = tf.shape(xseq)[1]
            xemb = source_embedding(xseq)
            initstate = tf.zeros([batch, shdim])

            annot = rnn_encoder(xemb, xmask, initstate)

            ymask = tf.ones([batch], dtype=tf.float32)
            istate, mannot = rnn_decoder.compute_initstate(annot)

            def sample_step(pemb, state, xmask, ymask, annot, mannot):
                alpha, context = rnn_decoder.compute_context(state, xmask,
                                                             annot, mannot)
                probs = rnn_decoder.compute_probability(pemb, state, context)
                logprobs = tf.log(probs)
                next_words = tf.multinomial(logprobs, 1)
                next_words = tf.squeeze(next_words, [1])
                yemb = target_embedding(next_words)
                next_state = rnn_decoder.compute_state(yemb, ymask, state,
                                                       context)
                next_words = tf.cast(next_words, tf.float32)
                return [next_words, yemb, next_state]

            iemb = tf.zeros([batch, tedim], tf.float32)

            seqs = []
            outputs_info = [None, iemb, istate]
            nonseqs = [xmask, ymask, annot, mannot]

            [words, embs, states] = scan(sample_step, seqs, outputs_info,
                                         nonseqs, n_steps=maxlen)
            words = tf.cast(words, tf.int32)

            return [xseq, xmask, maxlen], words

        mle_inputs, mle_outputs = mle_training_graph()
        mrt_inputs, mrt_outputs = mrt_training_graph()

        def build_function(graph_fn):
            inputs, outputs = graph_fn()
            return function(inputs, outputs)

        evaluate = build_function(mle_training_graph)

        functions = []
        functions.append(build_function(encoding_graph))
        functions.append(build_function(initstate_graph))
        functions.append(build_function(context_graph))
        functions.append(build_function(prediction_graph))
        functions.append(build_function(state_graph))

        def switch(mrt=False):
            if mrt:
                self.inputs = mrt_inputs
                self.outputs = mrt_outputs
                self.cost = mrt_outputs[0]
            else:
                self.cost = mle_outputs[0]
                self.inputs = mle_inputs
                self.outputs = mle_outputs

        self.name = scope
        self.config = config
        self.parameter = params
        self.option = option
        self.cost = mle_outputs[0]
        self.inputs = mle_inputs
        self.outputs = mle_outputs
        self.updates = []
        self.switch = switch
        self.search = functions
        self.evaluate = evaluate
        self.sampler = build_function(sampling_graph)
        self.attention = build_function(attention_graph)


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
    states = [None for i in range(num_models)]
    probs = [None for i in range(num_models)]

    xmask = np.ones(xseq.shape, "float32")

    for i in range(num_models):
        encode = functions[i][0]
        compute_istate = functions[i][1]
        annot[i] = encode(xseq, xmask)
        states[i], mannot[i] = compute_istate(annot[i])

    hdim = states[0].shape[1]
    cdim = annot[0].shape[2]
    # [num_models, batch, dim]
    states = np.array(states)

    trans = [[]]
    costs = [0.0]
    final_trans = []
    final_costs = []

    for k in range(maxlen):
        if size == 0:
            break

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
            alpha, contexts[i] = compute_context(states[i], xmasks, annots[i],
                                          mannots[i])

        for i in range(num_models):
            compute_probs = functions[i][3]
            probs[i] = compute_probs(last_words, states[i], contexts[i])

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
        newstates = np.zeros((num_models, size, hdim), "float32")
        newcontexts = np.zeros((num_models, size, cdim), "float32")
        inputs = np.zeros(size, "int32")

        for i, (idx, nword, ncost) in enumerate(zip(tinds, winds, costs)):
            newtrans[i] = trans[idx] + [nword]
            newcosts[i] = ncost
            for j in range(num_models):
                newstates[j][i] = states[j][idx]
                newcontexts[j][i] = contexts[j][idx]
            inputs[i] = nword

        ymask = np.ones((size,), "float32")

        for i in range(num_models):
            compute_state = functions[i][-1]
            newstates[i] = compute_state(inputs, ymask, newstates[i],
                                         newcontexts[i])

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

        states = newstates[:, indices]

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
