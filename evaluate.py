import os
import json
import utils
import tensorflow as tf

from inputs.input_function import multi_gpu_input_fn
from functions import get_model_fn


tf.logging.set_verbosity(tf.logging.INFO)


def main():
    def eval_input_fn():
        return multi_gpu_input_fn(data_dir, params['data'], params['n_devices'], params['batch_size'])

    params = utils.load_config_from_environ()
    print('==========Config=============>')
    print(json.dumps(params, indent=2))
    print('<=============================')
    run_config = tf.estimator.RunConfig(session_config=tf.ConfigProto(**params['sess_config']))
    data_dir = os.path.expanduser('~/datasets/cifar10')
    model_fn = get_model_fn(params['n_devices'], params['device'])
    estimator = tf.estimator.Estimator(model_fn,
                                       model_dir=params['model_dir'],
                                       params=params,
                                       config=run_config)

    for i, ckpt in enumerate(utils.generate_new_ckpt(params['model_dir'], wait_secs=120)):
        print('Evaluating `%s`' % ckpt)
        estimator.evaluate(eval_input_fn, checkpoint_path=ckpt)


if __name__ == '__main__':
    main()
