import tensorflow as tf
import tensorflow.contrib.slim as slim

from models import base

RESNET = {
    18: [2, 2, 2, 2],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}


class ResNet(base.BaseNetwork):
    def __init__(self, n_layers, n_classes):
        super().__init__(n_classes, name='resnet-%d' % n_layers)
        self.n_layers = n_layers

    def __call__(self, features, labels, mode, params):
        self.on_call(features, labels, mode, params)
        out = features
        blocks = RESNET[self.n_layers]
        self.in_planes = 64
        with tf.variable_scope(self.name):
            with slim.arg_scope([slim.batch_norm], is_training=self.train_mode()),\
                    base.conv2d_arg_scope(self.params['wd']):
                out = slim.conv2d(out, 64, 3, 1)
                out = self.stage_fn(out, 64, blocks[0], 1, 'stage-1')
                out = self.stage_fn(out, 128, blocks[1], 2, 'stage-2')
                out = self.stage_fn(out, 256, blocks[2], 2, 'stage-3')
                out = self.stage_fn(out, 512, blocks[3], 2, 'stage-4')

                out = tf.reduce_mean(out, axis=(1, 2), name='GAP')
                out = slim.fully_connected(out, self.n_classes, activation_fn=None)
        return out

    def bottleneck(self, x, in_dim, planes, stride=1):
        out = slim.conv2d(x, planes, 1, 1)
        out = slim.conv2d(out, planes, 3, stride)
        out = slim.conv2d(out, planes * 4, 1, 1, activation_fn=None, normalizer_fn=None)

        if stride != 1 or in_dim != planes * 4:
            shortcut = slim.conv2d(x, planes * 4, 1, stride, activation_fn=None, normalizer_fn=None,
                                   scope='shortcut')
        else:
            shortcut = x
        out = slim.batch_norm(out + shortcut, activation_fn=tf.nn.relu)
        return out

    def stage_fn(self, x, planes, n_blocks, stride, scope='stage'):
        out = x
        with tf.variable_scope(scope):
            for k in range(n_blocks):
                with tf.variable_scope('unit-%d' % k):
                    out = self.bottleneck(out, self.in_planes, planes, stride if k == 0 else 1)
                    self.in_planes = planes * 4
        return out


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, (2, 32, 32, 3))
    net = ResNet(18, 10)
    y = net(x, None, tf.estimator.ModeKeys.TRAIN, params={'wd': 4e-5})
    print(y.shape)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter('/tmp/resnet', sess.graph)
