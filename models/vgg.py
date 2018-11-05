import tensorflow as tf
import tensorflow.contrib.slim as slim

from .base import BaseNetwork


cfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

batch_norm_params = {
    'decay': 0.997,
    'epsilon': 1e-5,
    'scale': True,
    'updates_collections': tf.GraphKeys.UPDATE_OPS,
    'fused': None,
}


class VGG(BaseNetwork):
    def __init__(self, n_layers, n_classes):
        super().__init__(n_classes, name='vgg-%d' % n_layers)
        self.n_layers = n_layers

    def __call__(self, features, labels, mode, params):
        def _conv2d(_inx, _n):
            return slim.conv2d(_inx,
                               num_outputs=_n,
                               kernel_size=3,
                               padding='SAME',
                               weights_regularizer=slim.l2_regularizer(params['wd']),
                               activation_fn=tf.nn.relu,
                               normalizer_fn=slim.batch_norm,
                               normalizer_params=batch_norm_params)

        self.on_call(features, labels, mode, params)
        out = features
        with tf.variable_scope(self.name):
            with slim.arg_scope([slim.batch_norm], is_training=self.train_mode()):
                for x in cfg[self.n_layers]:
                    if x == 'M':
                        out = slim.max_pool2d(out, kernel_size=2, stride=2, padding='SAME')
                    else:
                        out = _conv2d(out, x)
                out = tf.reduce_mean(out, axis=(1, 2), name='GAP')
                out = slim.fully_connected(out, self.n_classes, activation_fn=None)
                return out
