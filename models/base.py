import tensorflow as tf
import tensorflow.contrib.slim as slim


class BaseNetwork:
    def __init__(self, n_classes, name):
        self.n_classes = n_classes
        self.name = name

    def on_call(self, features, labels, mode, params):
        self.features = features
        self.labels = labels
        self.mode = mode
        self.params = params

    def train_mode(self):
        return tf.estimator.ModeKeys.TRAIN == self.mode

    def eval_mode(self):
        return tf.estimator.ModeKeys.EVAL == self.mode

    def pred_mode(self):
        return tf.estimator.ModeKeys.PREDICT == self.mode


BATCH_NORM_PARAMS = {
    'decay': 0.997,
    'epsilon': 1e-5,
    'scale': True,
    'updates_collections': tf.GraphKeys.UPDATE_OPS,
    'fused': None,
}


def conv2d_arg_scope(wd):
    return slim.arg_scope([slim.conv2d],
                          padding='SAME',
                          weights_regularizer=slim.l2_regularizer(wd),
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=BATCH_NORM_PARAMS,
                          activation_fn=tf.nn.relu)
