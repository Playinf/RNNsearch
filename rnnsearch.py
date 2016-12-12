#!/usr/bin/python
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import os
import sys
import math
import time
import cPickle
import argparse

import numpy as np
import tensorflow as tf

from metric import bleu
from optimizer import optimizer
from model.rnnsearch import rnnsearch, beamsearch, batchsample
from data import textreader, textiterator, processdata, getlen


def loadvocab(file):
    fd = open(file, "r")
    vocab = cPickle.load(fd)
    fd.close()
    return vocab


def invertvoc(vocab):
    v = {}
    for k, idx in vocab.iteritems():
        v[idx] = k

    return v


def get_variable_values(var_list):
    session = tf.get_default_session()
    return [v.eval(session) for v in var_list]


def set_variables(var_list, val_list):
    updates = []

    for var, val in zip(var_list, val_list):
        updates.append(tf.assign(var, val))

    session = tf.get_default_session()

    session.run(tf.group(*updates))


def uniform(params, lower, upper, dtype="float32"):
    ops = []

    for var in params:
        s = var.get_shape().as_list()
        v = np.random.uniform(lower, upper, s).astype(dtype)
        ops.append(var.assign(v))

    session = tf.get_default_session()
    session.run(tf.group(*ops))


def parameters(params):
    n = 0

    for var in params:
        size = np.prod(var.get_shape().as_list())
        n += size

    return n


def serialize(name, model):
    fd = open(name, "w")
    option = model.option
    params = model.parameter

    cPickle.dump(option, fd)

    names = [var.name for var in params]
    vals = get_variable_values(params)
    pval = dict(list(zip(names, vals)))

    cPickle.dump(names, fd)
    np.savez(fd, **pval)
    fd.close()


# load model from file
def loadmodel(name):
    fd = open(name, "r")
    option = cPickle.load(fd)
    names = cPickle.load(fd)
    pval = dict(np.load(fd))

    params = [pval[n] for n in names]

    return option, params


def loadreferences(names, case=True):
    references = []
    reader = textreader(names)
    stream = textiterator(reader, size=[1, 1])

    for data in stream:
        newdata= []
        for batch in data:
            line = batch[0]
            words = line.strip().split()
            if not case:
                lower = [word.lower() for word in words]
                newdata.append(lower)
            else:
                newdata.append(words)

        references.append(newdata)

    stream.close()

    return references


def validate(scorpus, tcorpus, model, batch):

    if not scorpus or not tcorpus:
        return None

    reader = textreader([scorpus, tcorpus])
    stream = textiterator(reader, [batch, batch])
    svocabs, tvocabs = model.vocabulary
    unk_sym = model.option["unk"]
    eos_sym = model.option["eos"]

    totcost = 0.0
    count = 0

    for data in stream:
        xdata, xmask = processdata(data[0], svocabs[0], unk_sym, eos_sym)
        ydata, ymask = processdata(data[1], tvocabs[0], unk_sym, eos_sym)
        cost = model.compute(xdata, xmask, ydata, ymask)
        cost = cost[0]
        cost = cost * ymask.shape[1] / ymask.sum()
        totcost += cost / math.log(2)
        count = count + 1

    stream.close()

    bpc = totcost / count

    return bpc


def translate(model, corpus, **opt):
    fd = open(corpus, "r")
    svocab = model.option["vocabulary"][0][0]
    unk_sym = model.option["unk"]
    eos_sym = model.option["eos"]

    trans = []

    for line in fd:
        line = line.strip()
        data, mask = processdata([line], svocab, unk_sym, eos_sym)
        hls = beamsearch(model, data, **opt)
        if len(hls) > 0:
            best, score = hls[0]
            trans.append(best[:-1])
        else:
            trans.append([])

    fd.close()

    return trans


# format: source target prob
def load_dictionary(filename):
    fd = open(filename)

    mapping = {}

    for line in fd:
        sword, tword, prob = line.strip().split()
        prob = float(prob)

        if sword in mapping:
            oldword, oldprob = mapping[sword]
            if prob > oldprob:
                mapping[sword] = (tword, prob)
        else:
            mapping[sword] = (tword, prob)

    newmapping = {}
    for item in mapping:
        newmapping[item] = mapping[item][0]

    fd.close()

    return newmapping


def build_sample_space(refs, examples):
    space = {}

    for ref in refs:
        space[ref] = 1

    for example in examples:
        # remove empty
        if len(example) == 0:
            continue

        example = " ".join(example)

        if example in space:
            continue

        space[example] = 1

    return list(space.iterkeys())


def get_device(devid):
    if devid >= 0:
        return "/gpu:0"
    else:
        return "/cpu:0"


def parseargs_train(args):
    msg = "training rnnsearch"
    usage = "rnnsearch.py train [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    # corpus and vocabulary
    msg = "source and target corpus"
    parser.add_argument("--corpus", nargs=2, help=msg)
    msg = "source and target vocabulary"
    parser.add_argument("--vocab", nargs=2, help=msg)
    msg = "model name to save or saved model to initalize, required"
    parser.add_argument("--model", required=True, help=msg)

    # model parameters
    msg = "source and target embedding size, default 620"
    parser.add_argument("--embdim", nargs=2, type=int, help=msg)
    msg = "source, target and alignment hidden size, default 1000"
    parser.add_argument("--hidden", nargs=3, type=int, help=msg)
    msg = "maxout hidden dimension, default 500"
    parser.add_argument("--maxhid", type=int, help=msg)
    msg = "maxout number, default 2"
    parser.add_argument("--maxpart", type=int, help=msg)
    msg = "deepout hidden dimension, default 620"
    parser.add_argument("--deephid", type=int, help=msg)
    msg = "maximum training epoch, default 5"
    parser.add_argument("--maxepoch", type=int, help=msg)

    # tuning options
    msg = "learning rate, default 5e-4"
    parser.add_argument("--alpha", type=float, help=msg)
    msg = "momentum, default 0.0"
    parser.add_argument("--momentum", type=float, help=msg)
    msg = "batch size, default 128"
    parser.add_argument("--batch", type=int, help=msg)
    msg = "optimizer, default rmsprop"
    parser.add_argument("--optimizer", type=str, help=msg)
    msg = "gradient clipping, default 1.0"
    parser.add_argument("--norm", type=float, help=msg)
    msg = "early stopping iteration, default 0"
    parser.add_argument("--stop", type=int, help=msg)
    msg = "decay factor, default 0.5"
    parser.add_argument("--decay", type=float, help=msg)

    # validation
    msg = "random seed, default 1234"
    parser.add_argument("--seed", type=int, help=msg)
    msg = "compute bit per cost on validate dataset"
    parser.add_argument("--bpc", action="store_true", help=msg)
    msg = "validate dataset"
    parser.add_argument("--validate", type=str, help=msg)
    msg = "reference data"
    parser.add_argument("--ref", type=str, nargs="+", help=msg)

    # data processing
    msg = "sort batches"
    parser.add_argument("--sort", type=int, help=msg)
    msg = "shuffle every epcoh"
    parser.add_argument("--shuffle", type=int, help=msg)
    msg = "source and target sentence limit, default 50 (both), 0 to disable"
    parser.add_argument("--limit", type=int, nargs='+', help=msg)


    # control frequency
    msg = "save frequency, default 1000"
    parser.add_argument("--freq", type=int, help=msg)
    msg = "sample frequency, default 50"
    parser.add_argument("--sfreq", type=int, help=msg)
    msg = "validate frequency, default 1000"
    parser.add_argument("--vfreq", type=int, help=msg)

    # control beamsearch
    msg = "beam size, default 10"
    parser.add_argument("--beamsize", type=int, help=msg)
    msg = "normalize probability by the length of cadidate sentences"
    parser.add_argument("--normalize", type=int, help=msg)
    msg = "max translation length"
    parser.add_argument("--maxlen", type=int, help=msg)
    msg = "min translation length"
    parser.add_argument("--minlen", type=int, help=msg)

    # mrt training
    msg = "criterion, mle or mrt"
    parser.add_argument("--criterion", type=str, help=msg)
    msg = "sample space size"
    parser.add_argument("--sample", type=int, help=msg)
    msg = "sharpness parameter"
    parser.add_argument("--sharp", type=float, help=msg)

    # running device
    msg = "gpu id, -1 to use cpu"
    parser.add_argument("--gpuid", type=int, default=0, help=msg)

    msg = "reset count"
    parser.add_argument("--reset", type=int, help=msg)

    return parser.parse_args(args)


def parseargs_decode(args):
    msg = "translate using exsiting nmt model"
    usage = "rnnsearch.py translate [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)


    msg = "trained model"
    parser.add_argument("--model", nargs="+", required=True, help=msg)
    msg = "beam size"
    parser.add_argument("--beamsize", default=10, type=int, help=msg)
    msg = "normalize probability by the length of cadidate sentences"
    parser.add_argument("--normalize", action="store_true", help=msg)
    msg = "use arithmetic mean instead of geometric mean"
    parser.add_argument("--arithmetic", action="store_true", help=msg)
    msg = "max translation length"
    parser.add_argument("--maxlen", type=int, help=msg)
    msg = "min translation length"
    parser.add_argument("--minlen", type=int, help=msg)

    # running device
    msg = "gpu id, -1 to use cpu"
    parser.add_argument("--gpuid", type=int, default=0, help=msg)

    return parser.parse_args(args)


def parseargs_sample(args):
    msg = "sample sentence from exsiting nmt model"
    usage = "rnnsearch.py sample [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    # input model
    msg = "trained model"
    parser.add_argument("--model", required=True, help=msg)
    # batch size
    msg = "sample batch examples"
    parser.add_argument("--batch", default=1, type=int, help=msg)
    # max length
    msg = "max translation length"
    parser.add_argument("--maxlen", type=int, help=msg)

    # running device
    msg = "gpu id, -1 to use cpu"
    parser.add_argument("--gpuid", type=int, default=0, help=msg)

    return parser.parse_args(args)


def parseargs_replace(args):
    msg = "translate using exsiting nmt model"
    usage = "rnnsearch.py replace [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    msg = "trained models"
    parser.add_argument("--model", required=True, nargs="+", help=msg)
    msg = "source text and translation file"
    parser.add_argument("--text", required=True, nargs=2, help=msg)
    msg = "replacement dictionary"
    parser.add_argument("--dictionary", required=True, type=str, help=msg)
    msg = "replacement heuristic (0: copy, 1: replace, 2: heuristic replace)"
    parser.add_argument("--heuristic", type=int, default=1, help=msg)
    msg = "batch size"
    parser.add_argument("--batch", type=int, default=128, help=msg)
    msg = "use arithmetic mean instead of geometric mean"
    parser.add_argument("--arithmetic", action="store_true", help=msg)

    # running device
    msg = "gpu id, -1 to use cpu"
    parser.add_argument("--gpuid", type=int, default=0, help=msg)

    return parser.parse_args(args)


# default options
def getoption():
    option = {}

    # training corpus and vocabulary
    option["corpus"] = None
    option["vocab"] = None

    # model parameters
    option["embdim"] = [620, 620]
    option["hidden"] = [1000, 1000, 1000]
    option["maxpart"] = 2
    option["maxhid"] = 500
    option["deephid"] = 620

    # tuning options
    option["alpha"] = 5e-4
    option["batch"] = 128
    option["momentum"] = 0.0
    option["optimizer"] = "rmsprop"
    option["variant"] = "graves"
    option["norm"] = 1.0
    option["stop"] = 0
    option["decay"] = 0.5

    # runtime information
    option["cost"] = 0
    # batch count/reader count
    option["count"] = [0, 0]
    option["epoch"] = 0
    option["maxepoch"] = 5
    option["sort"] = 20
    option["shuffle"] = False
    option["limit"] = [50, 50]
    option["freq"] = 1000
    option["vfreq"] = 1000
    option["sfreq"] = 50
    option["seed"] = 1234
    option["validate"] = None
    option["ref"] = None
    option["bleu"] = 0.0

    # beam search
    option["beamsize"] = 10
    option["normalize"] = False
    option["maxlen"] = None
    option["minlen"] = None

    # special symbols
    option["unk"] = "UNK"
    option["eos"] = "<eos>"

    # criterion
    option["criterion"] = "mle"
    option["sample"] = 100
    option["sharp"] = 5e-3

    return option


def override_if_not_none(option, args, key):
    value = args.__dict__[key]
    option[key] = value if value != None else option[key]


# override default options
def override(option, args):

    # training corpus
    if args.corpus == None and option["corpus"] == None:
        raise ValueError("error: no training corpus specified")

    # vocabulary
    if args.vocab == None and option["vocab"] == None:
        raise ValueError("error: no training vocabulary specified")

    if args.limit and len(args.limit) > 2:
        raise ValueError("error: invalid number of --limit argument (<=2)")

    if args.limit and len(args.limit) == 1:
        args.limit = args.limit * 2

    override_if_not_none(option, args, "corpus")

    # vocabulary and model paramters cannot be overrided
    if option["vocab"] == None:
        option["vocab"] = args.vocab
        svocab = loadvocab(args.vocab[0])
        tvocab = loadvocab(args.vocab[1])
        isvocab = invertvoc(svocab)
        itvocab = invertvoc(tvocab)

        # compatible with groundhog
        option["source_eos_id"] = len(isvocab)
        option["target_eos_id"] = len(itvocab)

        svocab[option["eos"]] = option["source_eos_id"]
        tvocab[option["eos"]] = option["target_eos_id"]
        isvocab[option["source_eos_id"]] = option["eos"]
        itvocab[option["target_eos_id"]] = option["eos"]

        option["vocabulary"] = [[svocab, isvocab], [tvocab, itvocab]]

        # model parameters
        override_if_not_none(option, args, "embdim")
        override_if_not_none(option, args, "hidden")
        override_if_not_none(option, args, "maxhid")
        override_if_not_none(option, args, "maxpart")
        override_if_not_none(option, args, "deephid")

    # training options
    override_if_not_none(option, args, "maxepoch")
    override_if_not_none(option, args, "alpha")
    override_if_not_none(option, args, "momentum")
    override_if_not_none(option, args, "batch")
    override_if_not_none(option, args, "optimizer")
    override_if_not_none(option, args, "norm")
    override_if_not_none(option, args, "stop")
    override_if_not_none(option, args, "decay")

    # runtime information
    override_if_not_none(option, args, "validate")
    override_if_not_none(option, args, "ref")
    override_if_not_none(option, args, "freq")
    override_if_not_none(option, args, "vfreq")
    override_if_not_none(option, args, "sfreq")
    override_if_not_none(option, args, "seed")
    override_if_not_none(option, args, "sort")
    override_if_not_none(option, args, "shuffle")
    override_if_not_none(option, args, "limit")

    # beamsearch
    override_if_not_none(option, args, "beamsize")
    override_if_not_none(option, args, "normalize")
    override_if_not_none(option, args, "maxlen")
    override_if_not_none(option, args, "minlen")

    # training
    override_if_not_none(option, args, "criterion")
    override_if_not_none(option, args, "sample")
    override_if_not_none(option, args, "sharp")


def print_option(option):
    isvocab = option["vocabulary"][0][1]
    itvocab = option["vocabulary"][1][1]

    print ""
    print "options"

    print "corpus:", option["corpus"]
    print "vocab:", option["vocab"]
    print "vocabsize:", [len(isvocab), len(itvocab)]

    print "embdim:", option["embdim"]
    print "hidden:", option["hidden"]
    print "maxhid:", option["maxhid"]
    print "maxpart:", option["maxpart"]
    print "deephid:", option["deephid"]

    print "maxepoch:", option["maxepoch"]
    print "alpha:", option["alpha"]
    print "momentum:", option["momentum"]
    print "batch:", option["batch"]
    print "optimizer:", option["optimizer"]
    print "norm:", option["norm"]
    print "stop:", option["stop"]
    print "decay:", option["decay"]

    print "validate:", option["validate"]
    print "ref:", option["ref"]
    print "freq:", option["freq"]
    print "vfreq:", option["vfreq"]
    print "sfreq:", option["sfreq"]
    print "seed:", option["seed"]
    print "sort:", option["sort"]
    print "shuffle:", option["shuffle"]
    print "limit:", option["limit"]

    print "beamsize:", option["beamsize"]
    print "normalize:", option["normalize"]
    print "maxlen:", option["maxlen"]
    print "minlen:", option["minlen"]

    # training criterion
    print "criterion:", option["criterion"]
    print "sample:", option["sample"]
    print "sharp:", option["sharp"]

    # special symbols
    print "unk:", option["unk"]
    print "eos:", option["eos"]


def skipstream(stream, count):
    for i in range(count):
        stream.next()


def getfilename(name):
    s = name.split(".")
    return s[0]


def train(args):
    option = getoption()

    if os.path.exists(args.model):
        option, values = loadmodel(args.model)
        init = False
    else:
        init = True

    override(option, args)
    print_option(option)

    if option["ref"]:
        references = loadreferences(option["ref"])
    else:
        references = None

    svocabs, tvocabs = option["vocabulary"]
    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs

    pathname, basename = os.path.split(args.model)
    modelname = getfilename(basename)
    bestname = os.path.join(pathname, modelname + ".best.pkl")
    autoname = os.path.join(pathname, modelname + ".autosave.pkl")

    criterion = option["criterion"]

    batch = option["batch"] if criterion == "mle" else 1
    sortk = option["sort"] or 1 if criterion == "mle" else 1
    shuffle = option["seed"] if option["shuffle"] else None

    reader = textreader(option["corpus"], shuffle)
    processor = [getlen, getlen]
    stream = textiterator(reader, [batch, batch * sortk], processor,
                          option["limit"], option["sort"])

    if shuffle and "indices" in option and option["indices"] is not None:
        reader.set_indices(option["indices"])

    # reset to file beginning
    if args.reset:
        option["count"] = [0, 0]
        option["epoch"] = 0
        option["cost"] = 0.0

    skipstream(reader, option["count"][1])
    epoch = option["epoch"]
    maxepoch = option["maxepoch"]
    option["model"] = "rnnsearch"

    # tuning option
    toption = {}
    toption["algorithm"] = option["optimizer"]
    toption["variant"] = option["variant"]
    toption["constraint"] = ("norm", option["norm"])
    toption["norm"] = True

    alpha = option["alpha"]

    # beamsearch option
    doption = {}
    doption["beamsize"] = option["beamsize"]
    doption["normalize"] = option["normalize"]
    doption["maxlen"] = option["maxlen"]
    doption["minlen"] = option["minlen"]

    # create graph
    with tf.device(get_device(args.gpuid)):
        model = rnnsearch(**option)

        if criterion == "mrt":
            model.switch(mrt=True)

        trainer = optimizer(model, **toption)

    # create session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if args.gpuid >= 0:
        config.gpu_options.visible_device_list = "%d" % args.gpuid

    with tf.Session(config=config):
        print "parameters:", parameters(model.parameter)

        # set seed
        np.random.seed(option["seed"])
        tf.set_random_seed(option["seed"])

        tf.initialize_all_variables().run()

        if init:
            uniform(model.parameter, -0.08, 0.08)
        else:
            set_variables(model.parameter, values)

        unk_sym = option["unk"]
        eos_sym = option["eos"]
        best_score = option["bleu"]
        count = option["count"][0]
        totcost = option["cost"]

        for i in range(epoch, maxepoch):
            for data in stream:
                xdata, xmask = processdata(data[0], svocab, unk_sym, eos_sym)
                ydata, ymask = processdata(data[1], tvocab, unk_sym, eos_sym)

                if criterion == "mrt":
                    refs = []

                    for item in data[1]:
                        item = item.split()
                        item = [unk_sym if word not in tvocab else word
                                for word in item]
                        refs.append(" ".join(item))

                    t1 = time.time()

                    # sample from model
                    nsample = option["sample"] - len(refs)
                    xdata = np.repeat(xdata, nsample, 1)
                    xmask = np.repeat(xmask, nsample, 1)
                    examples = batchsample(model, xdata, xmask)
                    space = build_sample_space(refs, examples)
                    score = np.zeros((len(space),), "float32")

                    refs = [ref.split() for ref in refs]

                    for j in range(len(space)):
                        example = space[j].split()
                        score[j] = 1.0 - bleu([example], [refs], smooth=True)

                    ydata, ymask = processdata(space, tvocab, unk_sym, eos_sym)
                    cost, norm =  trainer.optimize(xdata[:, 0], ydata, ymask,
                                                   score, option["sharp"])
                    trainer.update(alpha=alpha)
                    t2 = time.time()

                    totcost += cost
                    count += 1
                    t = t2 - t1
                    ac = totcost / count
                    print i + 1, count, len(space), cost, norm, ac, t
                else:
                    t1 = time.time()
                    cost, norm = trainer.optimize(xdata, xmask, ydata, ymask)
                    trainer.update(alpha = alpha)
                    t2 = time.time()

                    count += 1
                    cost = cost * ymask.shape[1] / ymask.sum()
                    totcost += cost / math.log(2)
                    print i + 1, count, cost, norm, t2 - t1

                # save model
                if count % option["freq"] == 0:
                    model.option["bleu"] = best_score
                    model.option["cost"] = totcost
                    model.option["count"] = [count, reader.count]
                    model.option["indices"] = reader.get_indices()
                    serialize(autoname, model)

                # validate
                if count % option["vfreq"] == 0:
                    if option["validate"] and references:
                        trans = translate(model, option["validate"], **doption)
                        bleu_score = bleu(trans, references)
                        print "bleu: %2.4f" % bleu_score
                        if bleu_score > best_score:
                            best_score = bleu_score
                            model.option["cost"] = totcost
                            model.option["count"] = [count, reader.count]
                            model.option["bleu"] = best_score
                            model.option["indices"] = reader.get_indices()
                            serialize(bestname, model)

                # sampling and beamsearch
                if count % option["sfreq"] == 0:
                    n = len(data)
                    ind = np.random.randint(0, n)
                    sdata = data[0][ind]
                    tdata = data[1][ind]
                    xdata = xdata[:, ind:ind + 1]
                    xmask = xmask[:, ind:ind + 1]
                    sample = batchsample(model, xdata, xmask)
                    hypos = beamsearch(model, xdata)
                    best, score = hypos[0]
                    print sdata
                    print tdata
                    print " ".join(sample[0])
                    print " ".join(best[:-1])

            print "--------------------------------------------------"

            if option["validate"] and references:
                trans = translate(model, option["validate"], **doption)
                bleu_score = bleu(trans, references)
                print "iter: %d, bleu: %2.4f" % (i + 1, bleu_score)
                if bleu_score > best_score:
                    best_score = bleu_score
                    model.option["cost"] = totcost
                    model.option["count"] = [count, reader.count]
                    model.option["bleu"] = best_score
                    model.option["indices"] = reader.get_indices()
                    serialize(bestname, model)

            print "averaged cost: ", totcost / count
            print "--------------------------------------------------"

            # early stopping
            if i >= option["stop"]:
                alpha = alpha * option["decay"]

            count = 0
            totcost = 0
            stream.reset()

            model.option["epoch"] = i + 1
            model.option["alpha"] = alpha
            model.option["indices"] = reader.get_indices()
            model.option["bleu"] = best_score
            model.option["cost"] = totcost
            model.option["count"] = [0, 0]
            serialize(autoname, model)

        print "best(bleu): %2.4f" % best_score

    stream.close()


def decode(args):
    num_models = len(args.model)

    models = [None for i in range(num_models)]
    values = [None for i in range(num_models)]

    # create graph
    with tf.device(get_device(args.gpuid)):
        for i in range(num_models):
            option, values[i] = loadmodel(args.model[i])
            models[i] = rnnsearch(**option)

    # use the first model
    svocabs, tvocabs = models[0].option["vocabulary"]
    unk_symbol = models[0].option["unk"]
    eos_symbol = models[0].option["eos"]

    count = 0

    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs

    option = {}
    option["maxlen"] = args.maxlen
    option["minlen"] = args.minlen
    option["beamsize"] = args.beamsize
    option["normalize"] = args.normalize
    option["arithmetic"] = args.arithmetic

    # create session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if args.gpuid >= 0:
        config.gpu_options.visible_device_list = "%d" % args.gpuid

    with tf.Session(config=config):
        tf.initialize_all_variables().run()

        for i in range(num_models):
            set_variables(models[i].parameter, values[i])

        while True:
            line = sys.stdin.readline()

            if line == "":
                break

            data = [line]
            seq, mask = processdata(data, svocab, unk_symbol, eos_symbol)
            t1 = time.time()
            tlist = beamsearch(models, seq, **option)
            t2 = time.time()

            if len(tlist) == 0:
                sys.stdout.write("\n")
                score = -10000.0
            else:
                best, score = tlist[0]
                sys.stdout.write(" ".join(best[:-1]))
                sys.stdout.write("\n")

            count = count + 1
            sys.stderr.write(str(count) + " ")
            sys.stderr.write(str(score) + " " + str(t2 - t1) + "\n")


def sample(args):
    # create graph
    with tf.device(get_device(args.gpuid)):
        option, values = loadmodel(args.model)
        models = rnnsearch(**option)

    svocabs, tvocabs = models.option["vocabulary"]
    unk_symbol = models.option["unk"]
    eos_symbol = models.option["eos"]

    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs

    count = 0

    batch = args.batch

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if args.gpuid >= 0:
        config.gpu_options.visible_device_list = "%d" % args.gpuid

    with tf.Session(config=config):
        tf.initialize_all_variables().run()
        set_variables(models.parameter, values)

        while True:
            line = sys.stdin.readline()

            if line == "":
                break

            data = [line]
            seq, mask = processdata(data, svocab, unk_symbol, eos_symbol)
            t1 = time.time()
            seq = np.repeat(seq, batch, 1)
            mask = np.repeat(mask, batch, 1)
            tlist = batchsample(models, seq, mask, maxlen=args.maxlen)
            t2 = time.time()

            count = count + 1
            sys.stderr.write(str(count) + " ")

            if len(tlist) == 0:
                sys.stdout.write("\n")
            else:
                for i in range(min(args.batch, len(tlist))):
                    example = tlist[i]
                    sys.stdout.write(" ".join(example))
                    sys.stdout.write("\n")

            sys.stderr.write(str(t2 - t1) + "\n")


# unk replacement
def replace(args):
    num_models = len(args.model)

    # create graph
    models = [None for i in range(num_models)]
    values = [None for i in range(num_models)]

    with tf.device(get_device(args.gpuid)):
        for i in range(num_models):
            option, values[i] = loadmodel(args.model[i])
            models[i] = rnnsearch(**option)

    alignments = [None for i in range(num_models)]
    mapping = load_dictionary(args.dictionary)
    heuristic = args.heuristic

    # create session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if args.gpuid >= 0:
        config.gpu_options.visible_device_list = "%d" % args.gpuid

    with tf.Session(config=config), tf.device("/gpu:0"):
        tf.initialize_all_variables().run()

        for i in range(num_models):
            set_variables(models[i].parameter, values[i])

        # use the first model
        svocabs, tvocabs = models[0].option["vocabulary"]
        unk_sym = models[0].option["unk"]
        eos_sym = models[0].option["eos"]

        svocab, isvocab = svocabs
        tvocab, itvocab = tvocabs

        reader = textreader(args.text, False)
        stream = textiterator(reader, [args.batch, args.batch])

        for data in stream:
            xdata, xmask = processdata(data[0], svocab, unk_sym, eos_sym)
            ydata, ymask = processdata(data[1], tvocab, unk_sym, eos_sym)

            for i in range(num_models):
                # compute attention score
                alignments[i] = models[i].attention(xdata, xmask, ydata, ymask)

            # ensemble, alignment: yseq * xseq * batch
            if args.arithmetic:
                alignment = sum(alignments) / num_models
            else:
                alignments = map(np.log, alignments)
                alignment = np.exp(sum(alignments) / num_models)

            #  find source word to which each target word was most aligned
            indices = np.argmax(alignment, 1)

            # write to output
            for i in range(len(data[1])):
                source_words = data[0][i].strip().split()
                target_words = data[1][i].strip().split()
                translation = []

                for j in range(len(target_words)):
                    source_length = len(source_words)
                    word = target_words[j]

                    # found unk symbol
                    if word == unk_sym:
                        source_index = indices[j, i]

                        if source_index >= source_length:
                            translation.append(word)
                            continue

                        source_word = source_words[source_index]

                        if heuristic and source_word in mapping:
                            if heuristic == 1:
                                translation.append(mapping[source_word])
                            else:
                                # source word begin with lower case letter
                                if source_word.decode('utf-8')[0].islower():
                                    translation.append(mapping[source_word])
                                else:
                                    translation.append(source_word)
                        else:
                            translation.append(source_word)

                    else:
                        translation.append(word)

                sys.stdout.write(" ".join(translation))
                sys.stdout.write("\n")

    stream.close()


def helpinfo():
    print "usage:"
    print "\trnnsearch.py <command> [<args>]"
    print "use 'rnnsearch.py train --help' to see training options"
    print "use 'rnnsearch.py translate --help' to see translation options"
    print "use 'rnnsearch.py sample --help' to see sampling options"
    print "use 'rnnsearch.py replace --help' to see unk replacement options"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        helpinfo()
    else:
        command = sys.argv[1]
        if command == "train":
            print "training command:"
            print " ".join(sys.argv)
            args = parseargs_train(sys.argv[2:])
            train(args)
        elif command == "translate":
            sys.stderr.write(" ".join(sys.argv))
            sys.stderr.write("\n")
            args = parseargs_decode(sys.argv[2:])
            decode(args)
        elif command == "sample":
            sys.stderr.write(" ".join(sys.argv))
            sys.stderr.write("\n")
            args = parseargs_sample(sys.argv[2:])
            sample(args)
        elif command == "replace":
            sys.stderr.write(" ".join(sys.argv))
            sys.stderr.write("\n")
            args = parseargs_replace(sys.argv[2:])
            replace(args)
        else:
            helpinfo()
