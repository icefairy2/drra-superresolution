import tensorflow as tf
import numpy as np
import math

from keras.legacy import interfaces
import keras.backend as K
from keras.optimizers import Optimizer

def Conv(name, x, filter_size, in_filters, out_filters, strides):
    with tf.compat.v1.variable_scope(name):
        n = 1 / np.sqrt(filter_size * filter_size * in_filters)
        kernel = tf.compat.v1.get_variable('filter', [filter_size, filter_size, in_filters, out_filters], tf.compat.v1.float32,
                                 initializer=tf.compat.v1.random_uniform_initializer(minval=-n, maxval=n))
        bias = tf.compat.v1.get_variable('bias', [out_filters], tf.compat.v1.float32,
                               initializer=tf.compat.v1.random_uniform_initializer(minval=-n, maxval=n))

        return tf.compat.v1.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME') + bias


def Mean_shifter(x, rgb_mean, sign, rgb_range=255):
    kernel = tf.compat.v1.constant(name="identity", shape=[1, 1, 3, 3], value=np.eye(3).reshape(1, 1, 3, 3), dtype=tf.compat.v1.float32)
    bias = tf.compat.v1.constant(name='bias', shape=[3], value=[ele * rgb_range * sign for ele in rgb_mean], dtype=tf.compat.v1.float32)

    return tf.compat.v1.nn.conv2d(x, kernel, [1, 1, 1, 1], padding="SAME") + bias


def Channel_attention(name, x, ratio, n_feats):
    _res = x

    x = tf.compat.v1.reduce_mean(x, axis=[1, 2], keep_dims=True)
    x = Conv(name + '_conv1', x, 1, n_feats, n_feats // ratio, 1)
    x = tf.compat.v1.nn.relu(x)

    x = Conv(name + '_conv2', x, 1, n_feats // ratio, n_feats, 1)
    x = tf.nn.sigmoid(x)
    x = tf.compat.v1.multiply(x, _res)

    return x


def RCA_Block(name, x, filter_size, ratio, n_feats):
    _res = x

    x = Conv(name + '_conv1', x, filter_size, n_feats, n_feats, 1)
    x = tf.compat.v1.nn.relu(x)
    x = Conv(name + '_conv2', x, filter_size, n_feats, n_feats, 1)

    x = Channel_attention(name + '_CA', x, ratio, n_feats)

    x = x + _res

    return x


def Residual_Group(name, x, n_RCAB, filter_size, ratio, n_feats):
    skip_connection = x

    for i in range(n_RCAB):
        x = RCA_Block(name + '_%02d_RCAB' % i, x, filter_size, ratio, n_feats)

    x = Conv(name + '_conv_last', x, filter_size, n_feats, n_feats, 1)

    x = x + skip_connection

    return x


def Up_scaling(name, x, kernel_size, n_feats, scale):
    ## if scale is 2^n
    if (scale & (scale - 1) == 0):
        for i in range(int(math.log(scale, 2))):
            x = Conv(name + '_conv%02d' % i, x, kernel_size, n_feats, 2 * 2 * n_feats, 1)
            x = tf.compat.v1.depth_to_space(x, 2)

    elif scale == 3:
        x = Conv(name + '_conv', x, kernel_size, n_feats, 3 * 3 * n_feats, 1)
        x = tf.compat.v1.depth_to_space(x, 3)

    else:
        x = Conv(name + '_conv', x, kernel_size, scale * scale * n_feats, 1)
        x = tf.compat.v1.depth_to_space(x, scale)

    return x



class Adam_lr_mult(Optimizer):
    """Adam optimizer.
    Adam optimizer, with learning rate multipliers built on Keras implementation
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)

    AUTHOR: Erik Brorson
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False,
                 multipliers=None, debug_verbose=False, **kwargs):
        super(Adam_lr_mult, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.multipliers = multipliers
        self.debug_verbose = debug_verbose

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):

            # Learning rate multipliers
            if self.multipliers:
                multiplier = [mult for mult in self.multipliers if mult in p.name]
            else:
                multiplier = None
            if multiplier:
                new_lr_t = lr_t * self.multipliers[multiplier[0]]
                if self.debug_verbose:
                    print('Setting {} to learning rate {}'.format(multiplier[0], new_lr_t))
                    print(K.get_value(new_lr_t))
            else:
                new_lr_t = lr_t
                if self.debug_verbose:
                    print('No change in learning rate {}'.format(p.name))
                    print(K.get_value(new_lr_t))
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - new_lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - new_lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad,
                  'multipliers': self.multipliers}
        base_config = super(Adam_lr_mult, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))