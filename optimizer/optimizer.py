# optimizer.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import updates
import tensorflow as tf

from nn import function
from tensorflow.python.framework.ops import colocate_with


def create_zeros_slot(primary, name, dtype=None):
    if dtype is None:
        dtype = primary.dtype
    shape = primary.get_shape().as_list()
    init_val = tf.zeros_initializer(shape, dtype=dtype)
    with colocate_with(primary):
        var = tf.Variable(init_val, name=name, trainable=False)
    return var


class optimizer:

    def __init__(self, model, **option):
        loss = model.cost
        params = model.parameter
        inputs = model.inputs
        outputs = model.outputs

        if "norm" not in option:
            option["norm"] = False

        if "constraint" not in option:
            option["constraint"] = None

        grads = tf.gradients(loss, params, colocate_gradients_with_ops=True,
                             gate_gradients=True)

        if option["norm"]:
            normval = tf.global_norm(grads)
            outputs = outputs[:]
            outputs.append(normval)

        if option["constraint"]:
            method, value = option["constraint"]
            if method == "value":
                min_v = value[0]
                max_v = value[1]
                grads = [tf.clip_by_value(g, min_v, max_v) for g in grads]
            if method == "norm":
                grads, normval = tf.clip_by_global_norm(grads, value)

        gvars = []
        gvars_and_vars = []
        grads_and_gvars = []

        for grad, var in zip(grads, params):
            if grad is None:
                continue
            slotvar = create_zeros_slot(var, "gradient")
            gvars.append(slotvar)
            gvars_and_vars.append((slotvar, var))
            grads_and_gvars.append([grad, slotvar])

        grad_updates = updates.grad_updates(grads_and_gvars)
        placeholders = []

        if "algorithm" not in option:
            option["algorithm"] = "sgd"

        if option["algorithm"] == "sgd":
            varlist = []
            lr = tf.placeholder(tf.float32, [])
            defaults = [('alpha', 1.0)]
            placeholders.append(lr)
            var_updates = updates.sgd_updates(gvars_and_vars, lr)
        elif option["algorithm"] == "rmsprop":
            lr = tf.placeholder(tf.float32, [])
            rho = tf.placeholder(tf.float32, [])
            eps = tf.placeholder(tf.float32, [])
            varlist = []
            svars = []

            for gvar in gvars:
                ms = create_zeros_slot(gvar, "mean_square")
                mg = create_zeros_slot(gvar, "mean_gradient")
                svars.append([ms, mg])
                varlist.extend([ms, mg])

            placeholders.append(lr)
            placeholders.append(rho)
            placeholders.append(eps)
            defaults = [('alpha', 1e-2), ('rho', 0.99), ('epsilon', 1e-8)]
            var_updates = updates.rmsprop_updates(gvars_and_vars, svars,
                                                  lr, rho, eps)
        elif option["algorithm"] == "adam":
            lr = tf.placeholder(tf.float32, [])
            beta1 = tf.placeholder(tf.float32, [])
            beta2 = tf.placeholder(tf.float32, [])
            eps = tf.placeholder(tf.float32, [])

            t = tf.Variable(0.0, name="adam/t", dtype=tf.float32,
                            trainable=False)
            varlist = [t]
            svars = [t]

            for gvar in gvars:
                m = create_zeros_slot(gvar, "m")
                v = create_zeros_slot(gvar, "v")
                svars.append([m, v])
                varlist.extend([m, v])

            placeholders.append(lr)
            placeholders.append(beta1)
            placeholders.append(beta2)
            placeholders.append(eps)
            defaults = [("alpha", 1e-3), ("beta1", 0.9), ("beta2", 0.999),
                        ("epsilon", 1e-8)]
            var_updates = updates.adam_updates(gvars_and_vars, svars,
                                               lr, beta1, beta2, eps)
        else:
            raise ValueError("unknown algorithm %s" % option["algorithm"])

        optimize = function(inputs, outputs, updates=grad_updates)
        update = function(placeholders, [], updates=var_updates)

        def wrapper(**option):
            values = []
            for item in defaults:
                name = item[0]
                val = item[1]
                if name not in option:
                    option[name] = val
                values.append(option[name])
            return update(*values)

        self.optimize = optimize
        self.update = wrapper
        self.option = option
        self.variables = varlist
