# updates.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import tensorflow as tf


def sgd_update(grad, var, lr):
    delta = lr * grad
    return tf.assign_sub(var, delta)


def adam_update(grad, var, a_t, m, v, lr, beta1, beta2, eps):
    m_t = beta1 * m + (1 - beta1) * grad
    v_t = beta2 * v + (1 - beta2) * tf.square(grad)
    delta = a_t * m_t / (tf.sqrt(v_t) + eps)

    update_mt = tf.assign(m, m_t)
    update_vt = tf.assign(v, v_t)
    update_delta = tf.assign_sub(var, delta)

    return tf.group(update_mt, update_vt, update_delta)


def rmsprop_update(grad, var, ms, mg, lr, rho, eps):
    new_ms = rho * ms + (1.0 - rho) * tf.square(grad)
    new_mg = rho * mg + (1.0 - rho) * grad

    delta = lr * grad / tf.sqrt(new_ms - tf.square(new_mg) + eps)

    update_ms = tf.assign(ms, new_ms)
    update_mg = tf.assign(mg, new_mg)
    update_var = tf.assign_sub(var, delta)

    return tf.group(update_ms, update_mg, update_var)


def grad_updates(grads_and_vars):
    updates = []

    for grad, var in grads_and_vars:
        if isinstance(grad, tf.Tensor):
            updates.append(tf.assign(var, grad))
        else:
            new_var = tf.assign(var, tf.zeros_like(var))
            updates.append(tf.scatter_add(new_var, grad.indices, grad.values))

    return tf.group(*updates)


def sgd_updates(grads_and_vars, lr):
    updates = []
    for grad, var in grads_and_vars:
        updates.append(sgd_update(grad, var, lr))

    return tf.group(*updates)


def adam_updates(grads_and_vars, slot_vars, lr, beta1, beta2, eps):
    updates = []
    t = slot_vars[0]
    slot_vars = slot_vars[1:]

    new_t = t + 1
    a = lr * tf.sqrt(1 - tf.pow(beta2, new_t)) / (1 - tf.pow(beta1, new_t))

    updates.append(tf.assign(t, new_t))

    for gv, sv in zip(grads_and_vars, slot_vars):
        grad, var = gv
        m, v = sv
        updates.append(adam_update(grad, var, a, m, v, lr, beta1, beta2, eps))

    return tf.group(*updates)


def rmsprop_updates(grads_and_vars, slot_vars, lr, rho, eps):
    updates = []

    for gv, sv in zip(grads_and_vars, slot_vars):
        grad, var = gv
        ms, mg = sv
        updates.append(rmsprop_update(grad, var, ms, mg, lr, rho, eps))

    return tf.group(*updates)
