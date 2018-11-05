import os
import time
import config
import tensorflow as tf


def load_config_from_environ():
    default = config.CONFIG.copy()
    for key, value in default.items():
        default[key] = type(value)(os.environ[key]) if key in os.environ else value
    default['batch_size'] = default['n_devices'] * default['batch_size_per_device']
    return default


def generate_new_ckpt(model_dir, n_loops=None, wait_secs=60):
    old_ckpts = set()
    n_loops = n_loops or int(1e8)
    for _ in range(n_loops):
        ckpt_state = tf.train.get_checkpoint_state(model_dir)
        all_ckpts = set(ckpt_state.all_model_checkpoint_paths) if ckpt_state else set()
        new_ckpts = all_ckpts - old_ckpts
        if len(new_ckpts) == 0:
            print('Wait for %d seconds' % wait_secs)
            try:
                time.sleep(wait_secs)
            except KeyboardInterrupt:
                ans = input('Sure you wanna to exit?(y|n)')
                if ans.startswith('y'):
                    break
        else:
            yield from sorted(new_ckpts, key=lambda x: int(x.split('-')[-1]))
            old_ckpts = all_ckpts
