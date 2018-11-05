import itertools
import tensorflow as tf

from models import load_network


def get_model_fn(n_devices, device='gpu'):
    def _resnet_model_fn(features, labels, mode, params):
        tower_losses, tower_gradvars, tower_preds = [], [], []

        for i in range(n_devices):
            with tf.variable_scope(params['network'], reuse=bool(i != 0)):
                with tf.name_scope('tower_%d' % i) as name_scope:
                    with tf.device('/%s:%d' % (device, i)):
                        loss, gradvars, preds = _tower_fn(features[i], labels[i], mode, params)
                        tower_losses.append(loss)
                        tower_gradvars.append(gradvars)
                        tower_preds.append(preds)
                        if i == 0:
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                           name_scope)

        gradvars = []
        with tf.name_scope('gradient_averaging'):
            all_grads = {}
            for grad, var in itertools.chain(*tower_gradvars):
                if grad is not None:
                    all_grads.setdefault(var, []).append(grad)
            for var, grads in all_grads.items():
                with tf.device(var.device):
                    if len(grads) == 1:
                        avg_grad = grads[0]
                    else:
                        avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
                gradvars.append((avg_grad, var))
        with tf.device('/gpu:0'):
            loss = tf.reduce_mean(tower_losses, name='loss')
            reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            tf.summary.scalar('reg_loss', reg_loss)
            logging_hook = tf.train.LoggingTensorHook(tensors={'loss': loss}, every_n_iter=100)
            solver = tf.train.MomentumOptimizer(learning_rate=params['lr'],
                                                momentum=0.9)
            train_op = [solver.apply_gradients(gradvars,
                                               global_step=tf.train.get_or_create_global_step(),
                                               name='apply_grad')]
            train_op.extend(update_ops)
            train_op = tf.group(*train_op)
            predictions = {
                  'classes': tf.concat([p['classes'] for p in tower_preds], axis=0),
                  'probabilities': tf.concat([p['probabilities'] for p in tower_preds], axis=0)
            }
            stacked_labels = tf.concat(labels, axis=0)
            metrics = {
                'accuracy': tf.metrics.accuracy(stacked_labels, predictions['classes'])
            }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics,
            training_hooks=[logging_hook],
        )
    return _resnet_model_fn


def _tower_fn(features, labels, mode, params):
    model_fn = load_network(params['network'])(n_classes=params['n_classes'])
    logits = model_fn(features, labels, mode, params)
    tower_pred = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits),
    }
    data_loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
    data_loss = tf.reduce_mean(data_loss)
    reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    tower_loss = tf.add(data_loss, reg_loss, name='total_loss')
    model_params = tf.trainable_variables()
    tower_grad = tf.gradients(tower_loss, model_params)
    return tower_loss, zip(tower_grad, model_params), tower_pred
