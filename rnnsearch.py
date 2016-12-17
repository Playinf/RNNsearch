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

from bleu import bleu
from optimizer import optimizer
from model import rnnsearch, beamsearch
from data import textreader, textiterator


def data_length(line):
    return len(line.split())


def convert_data(data, voc, unk="UNK", eos="<eos>", time_major=True):
    # tokenize
    data = [line.split() + [eos] for line in data]

    unkid = voc[unk]

    newdata = []

    for d in data:
        idlist = [voc[w] if w in voc else unkid for w in d]
        newdata.append(idlist)

    data = newdata

    lens = [len(tokens) for tokens in data]

    n = len(lens)
    maxlen = np.max(lens)

    batch_data = np.zeros((n, maxlen), "int32")
    data_length = np.array(lens)

    for idx, item in enumerate(data):
        batch_data[idx, :lens[idx]] = item

    if time_major:
        batch_data = batch_data.transpose()

    return batch_data, data_length


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


def set_variables(var_list, value_list):
    session = tf.get_default_session()

    for var, val in zip(var_list, value_list):
        session.run(var.assign(val))


def uniform(params, lower, upper, dtype="float32"):
    val_list = []

    for var in params:
        s = var.get_shape().as_list()
        v = np.random.uniform(lower, upper, s).astype(dtype)
        val_list.append(v)

    set_variables(params, val_list)


def parameters(params):
    n = 0

    for var in params:
        size = np.prod(var.get_shape().as_list())
        n += size

    return n


def serialize(name, model):
    fd = open(name, "w")
    option = model.option
    cPickle.dump(option, fd)

    session = tf.get_default_session()
    params = tf.trainable_variables()

    name_list = []
    pval = {}

    for param in params:
        name_list.append(param.name)
        pval[param.name] = param.eval(session)

    cPickle.dump(name_list, fd)
    np.savez(fd, **pval)
    fd.close()


def loadmodel(name):
    fd = open(name, "r")
    option = cPickle.load(fd)
    name_list = cPickle.load(fd)

    params = np.load(fd)
    params = dict(params)
    params = [params[nstr] for nstr in name_list]

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


def translate(model, corpus, **opt):
    fd = open(corpus, "r")
    svocab = model.option["vocabulary"][0][0]
    unk_symbol = model.option["unk"]
    eos_symbol = model.option["eos"]

    trans = []

    for line in fd:
        line = line.strip()
        data, length = convert_data([line], svocab, unk_symbol, eos_symbol)
        hls = beamsearch(model, data, **opt)
        if len(hls) > 0:
            best, score = hls[0]
            trans.append(best[:-1])
        else:
            trans.append([])

    fd.close()

    return trans


def parseargs_train(args):
    msg = "training rnnsearch"
    usage = "rnnsearch.py train [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    msg = "source and target corpus"
    parser.add_argument("--corpus", nargs=2, help=msg)
    msg = "source and target vocabulary"
    parser.add_argument("--vocab", nargs=2, help=msg)
    msg = "model name to save or saved model to initalize, required"
    parser.add_argument("--model", required=True, help=msg)

    msg = "source and target embedding size, default 620"
    parser.add_argument("--embdim", type=int, help=msg)
    msg = "source, target and alignment hidden size, default 1000"
    parser.add_argument("--hidden", type=int, help=msg)
    msg = "attention hidden dimension, default 1000"
    parser.add_argument("--attention", type=int, help=msg)

    msg = "maximum training epoch, default 5"
    parser.add_argument("--maxepoch", type=int, help=msg)
    msg = "learning rate, default 5e-4"
    parser.add_argument("--alpha", type=float, help=msg)
    msg = "batch size, default 128"
    parser.add_argument("--batch", type=int, help=msg)
    msg = "optimizer, default adam"
    parser.add_argument("--optimizer", type=str, help=msg)
    msg = "gradient clipping, default 1.0"
    parser.add_argument("--norm", type=float, help=msg)
    msg = "early stopping iteration, default 0"
    parser.add_argument("--stop", type=int, help=msg)
    msg = "decay factor, default 0.5"
    parser.add_argument("--decay", type=float, help=msg)
    msg = "random seed, default 1234"
    parser.add_argument("--seed", type=int, help=msg)
    msg = "initialzing scale"
    parser.add_argument("--scale", type=float, help=msg)

    msg = "validation dataset"
    parser.add_argument("--validation", type=str, help=msg)
    msg = "reference data"
    parser.add_argument("--references", type=str, nargs="+", help=msg)

    # data processing
    msg = "sort batches"
    parser.add_argument("--sort", type=int, help=msg)
    msg = "shuffle every epcoh"
    parser.add_argument("--shuffle", type=int, help=msg)
    msg = "source and target sentence limit, default 50 (both), 0 to disable"
    parser.add_argument("--limit", type=int, nargs='+', help=msg)

    # save frequency
    msg = "save frequency, default 1000"
    parser.add_argument("--freq", type=int, help=msg)
    msg = "display frequency, default 50"
    parser.add_argument("--dfreq", type=int, help=msg)
    msg = "validation frequency, default 1000"
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

    # running device
    msg = "gpu id, -1 to use cpu"
    parser.add_argument("--gpuid", type=int, default=0, help=msg)

    msg = "reset"
    parser.add_argument("--reset", type=int, default=0, help=msg)

    return parser.parse_args(args)


def parseargs_decode(args):
    msg = "translate using exsiting model"
    usage = "rnnsearch.py translate [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    msg = "trained model"
    parser.add_argument("--model", required=True, help=msg)
    msg = "beam size"
    parser.add_argument("--beamsize", default=10, type=int, help=msg)
    msg = "normalize probability by the length of cadidate sentences"
    parser.add_argument("--normalize", action="store_true", help=msg)
    msg = "max translation length"
    parser.add_argument("--maxlen", type=int, help=msg)
    msg = "min translation length"
    parser.add_argument("--minlen", type=int, help=msg)

    # running device
    msg = "gpu id, -1 to use cpu"
    parser.add_argument("--gpuid", type=int, default=0, help=msg)

    return parser.parse_args(args)


# default options
def default_option():
    option = {}

    # training corpus and vocabulary
    option["corpus"] = None
    option["vocab"] = None

    # model parameters
    option["embdim"] = 620
    option["hidden"] = 1000
    option["attention"] = 1000

    # tuning options
    option["alpha"] = 5e-4
    option["batch"] = 128
    option["optimizer"] = "adam"
    option["norm"] = 5.0
    option["stop"] = 0
    option["decay"] = 0.5
    option["scale"] = 0.08

    # runtime information
    option["cost"] = 0.0
    option["count"] = [0, 0]
    option["epoch"] = 0
    option["maxepoch"] = 5
    option["sort"] = 20
    option["shuffle"] = False
    option["limit"] = [50, 50]
    option["freq"] = 1000
    option["vfreq"] = 1000
    option["dfreq"] = 50
    option["seed"] = 1234
    option["validation"] = None
    option["references"] = None
    option["bleu"] = 0.0
    option["indices"] = None

    # beam search
    option["beamsize"] = 10
    option["normalize"] = False
    option["maxlen"] = None
    option["minlen"] = None

    # special symbols
    option["unk"] = "UNK"
    option["eos"] = "</s>"

    return option


def args_to_dict(args):
    return args.__dict__


def override_if_not_none(option, newopt, key):
    if key not in newopt or key not in option:
        return
    if newopt[key]:
        option[key] = newopt[key]


# override default options
def override(option, newopt):

    # training corpus
    if newopt["corpus"] == None and option["corpus"] == None:
        raise RuntimeError("error: no training corpus specified")

    # vocabulary
    if newopt["vocab"] == None and option["vocab"] == None:
        raise RuntimeError("error: no training vocabulary specified")

    override_if_not_none(option, newopt, "corpus")

    # vocabulary and model paramters cannot be overrided
    if option["vocab"] == None:
        option["vocab"] = newopt["vocab"]

        svocab = loadvocab(args.vocab[0])
        tvocab = loadvocab(args.vocab[1])
        isvocab = invertvoc(svocab)
        itvocab = invertvoc(tvocab)

        # compatible with groundhog
        svocab[option["eos"]] = len(isvocab)
        tvocab[option["eos"]] = len(itvocab)
        isvocab[len(isvocab)] = option["eos"]
        itvocab[len(itvocab)] = option["eos"]

        option["vocabulary"] = [[svocab, isvocab], [tvocab, itvocab]]

        # model parameters
        override_if_not_none(option, newopt, "embdim")
        override_if_not_none(option, newopt, "hidden")
        override_if_not_none(option, newopt, "attention")

    # training options
    override_if_not_none(option, newopt, "maxepoch")
    override_if_not_none(option, newopt, "alpha")
    override_if_not_none(option, newopt, "batch")
    override_if_not_none(option, newopt, "optimizer")
    override_if_not_none(option, newopt, "norm")
    override_if_not_none(option, newopt, "stop")
    override_if_not_none(option, newopt, "decay")

    # runtime information
    override_if_not_none(option, newopt, "validation")
    override_if_not_none(option, newopt, "references")
    override_if_not_none(option, newopt, "freq")
    override_if_not_none(option, newopt, "vfreq")
    override_if_not_none(option, newopt, "dfreq")
    override_if_not_none(option, newopt, "seed")
    override_if_not_none(option, newopt, "sort")
    override_if_not_none(option, newopt, "shuffle")
    override_if_not_none(option, newopt, "limit")

    # beamsearch
    override_if_not_none(option, newopt, "beamsize")
    override_if_not_none(option, newopt, "normalize")
    override_if_not_none(option, newopt, "maxlen")
    override_if_not_none(option, newopt, "minlen")


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
    print "attention:", option["attention"]

    print "maxepoch:", option["maxepoch"]
    print "alpha:", option["alpha"]
    print "batch:", option["batch"]
    print "optimizer:", option["optimizer"]
    print "norm:", option["norm"]
    print "stop:", option["stop"]
    print "decay:", option["decay"]

    print "validation:", option["validation"]
    print "references:", option["references"]
    print "freq:", option["freq"]
    print "vfreq:", option["vfreq"]
    print "dfreq:", option["dfreq"]
    print "seed:", option["seed"]
    print "sort:", option["sort"]
    print "shuffle:", option["shuffle"]
    print "limit:", option["limit"]

    print "beamsize:", option["beamsize"]
    print "normalize:", option["normalize"]
    print "maxlen:", option["maxlen"]
    print "minlen:", option["minlen"]

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
    option = default_option()

    # load saved models
    if os.path.exists(args.model):
        option, values = loadmodel(args.model)
        init = False
    else:
        init = True

    override(option, args_to_dict(args))
    print_option(option)

    # prepare
    pathname, basename = os.path.split(args.model)
    modelname = getfilename(basename)
    autoname = os.path.join(pathname, modelname + ".autosave.pkl")
    bestname = os.path.join(pathname, modelname + ".best.pkl")
    batch = option["batch"]
    sortk = option["sort"] or 1
    shuffle = option["seed"] if option["shuffle"] else None
    reader = textreader(option["corpus"], shuffle)
    stream = textiterator(reader, [batch, batch * sortk],
                          [data_length, data_length],
                          option["limit"], option["sort"])

    if shuffle and "indices" in option and option["indices"] is not None:
        reader.set_indices(option["indices"])

    if args.reset:
        option["count"] = [0, 0]
        option["epoch"] = 0
        option["cost"] = 0.0

    skipstream(reader, option["count"][1])
    epoch = option["epoch"]
    maxepoch = option["maxepoch"]

    if option["references"]:
        references = loadreferences(option["references"])
    else:
        references = None

    alpha = option["alpha"]

    # beamsearch option
    bopt = {}
    bopt["beamsize"] = option["beamsize"]
    bopt["normalize"] = option["normalize"]
    bopt["maxlen"] = option["maxlen"]
    bopt["minlen"] = option["minlen"]

    # misc
    svocabs, tvocabs = option["vocabulary"]
    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs
    unk = option["unk"]
    eos = option["eos"]

    best_score = option["bleu"]
    epoch = option["epoch"]
    maxepoch = option["maxepoch"]

    # create session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if args.gpuid >= 0:
        config.gpu_options.visible_device_list = "%d" % args.gpuid

    with tf.Session(config=config):
        model = rnnsearch(option["embdim"], option["hidden"],
                          option["attention"], len(isvocab), len(itvocab))

        model.option = option
        print "parameters:", parameters(tf.trainable_variables())

        # set seed
        np.random.seed(option["seed"])
        tf.set_random_seed(option["seed"])

        # create optimizer
        optim = optimizer(model, algorithm=option["optimizer"],
                          norm=option["norm"])

        tf.global_variables_initializer().run()

        if init:
            uniform(tf.trainable_variables(), -0.08, 0.08)
        else:
            set_variables(tf.trainable_variables(), values)

        best_score = option["bleu"]
        count = option["count"][0]
        totcost = option["cost"]

        for i in range(epoch, maxepoch):
            for data in stream:
                xdata, xlen = convert_data(data[0], svocab, unk, eos)
                ydata, ylen = convert_data(data[1], tvocab, unk, eos)

                t1 = time.time()
                cost, norm = optim.optimize(xdata, xlen, ydata, ylen)
                optim.update(alpha=alpha)
                t2 = time.time()

                count += 1
                cost = cost * len(ylen) / sum(ylen)
                totcost += cost / math.log(2)

                print i + 1, count, cost, norm, t2 - t1

                # save model
                if count % option["freq"] == 0:
                    model.option["indices"] = reader.get_indices()
                    model.option["bleu"] = best_score
                    model.option["cost"] = totcost
                    model.option["count"] = [count, reader.count]
                    serialize(autoname, model)

                if count % option["vfreq"] == 0:
                    if option["validation"] and references:
                        trans = translate(model, option["validation"], **bopt)
                        bleu_score = bleu(trans, references)
                        print "bleu: %2.4f" % bleu_score
                        if bleu_score > best_score:
                            model.option["indices"] = reader.get_indices()
                            model.option["bleu"] = best_score
                            model.option["cost"] = totcost
                            model.option["count"] = [count, reader.count]
                            serialize(bestname, model)

                if count % option["dfreq"] == 0:
                    batch = len(data[0])
                    ind = np.random.randint(0, batch)
                    sdata = data[0][ind]
                    tdata = data[1][ind]
                    xdata = xdata[:, ind:ind + 1]
                    hls = beamsearch(model, xdata, **bopt)
                    best, score = hls[0]
                    print sdata
                    print tdata
                    print "search score:", score
                    print "translation:", " ".join(best[:-1])

            print "--------------------------------------------------"

            if option["vfreq"] and references:
                trans = translate(model, option["validation"], **bopt)
                bleu_score = bleu(trans, references)
                print "iter: %d, bleu: %2.4f" % (i + 1, bleu_score)
                if bleu_score > best_score:
                    best_score = bleu_score
                    model.option["indices"] = reader.get_indices()
                    model.option["bleu"] = best_score
                    model.option["cost"] = totcost
                    model.option["count"] = [count, reader.count]
                    serialize(bestname, model)

            print "averaged cost: ", totcost / option["count"]
            print "--------------------------------------------------"

            # early stopping
            if i >= option["stop"]:
                alpha = alpha * option["decay"]

            count = 0
            totcost = 0.0
            stream.reset()

            # update autosave
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
    option, values = loadmodel(args.model)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if args.gpuid >= 0:
        config.gpu_options.visible_device_list = "%d" % args.gpuid

    svocabs, tvocabs = option["vocabulary"]
    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs

    unk_sym = option["unk"]
    eos_sym = option["eos"]

    count = 0

    doption = {}
    doption["maxlen"] = args.maxlen
    doption["minlen"] = args.minlen
    doption["beamsize"] = args.beamsize
    doption["normalize"] = args.normalize

    with tf.Session(config=config):
        model = rnnsearch(option["embdim"], option["hidden"],
                          option["attention"], len(isvocab), len(itvocab))

        model.option = option

        tf.initialize_all_variables().run()
        set_variables(tf.trainable_variables(), values)

        while True:
            line = sys.stdin.readline()

            if line == "":
                break

            data = [line]
            seq, seq_len = convert_data(data, svocab, unk_sym, eos_sym)
            t1 = time.time()
            tlist = beamsearch(model, seq, **doption)
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


def helpinfo():
    print "usage:"
    print "\trnnsearch.py <command> [<args>]"
    print "using rnnsearch.py train --help to see training options"
    print "using rnnsearch.py translate --help to see translation options"


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
        else:
            helpinfo()
