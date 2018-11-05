CONFIG = {
    'data': 'trainval',
    'lr': 1e-1,
    'batch_size_per_device': 128,
    'network': 'vgg-13',
    'wd': 5e-4,
    'n_classes': 10,
    'device': 'gpu',
    'n_devices': 1,
    'save_every': 1000,
    'model_dir': '/tmp/cifar10',
    'n_epochs': 1,
    'sess_config': {
        'allow_soft_placement': True,
        'log_device_placement': False,
        'gpu_options': {'allow_growth': True},
    },
}
