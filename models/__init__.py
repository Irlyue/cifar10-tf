from .vgg import VGG
from .resnet import ResNet


def load_network(network):
    network = network.lower()
    if network.startswith('vgg'):
        _, n_layers = network.split('-')
        return lambda **kwargs: VGG(int(n_layers), **kwargs)
    elif network.startswith('resnet'):
        _, n_layers = network.split('-')
        return lambda **kwargs: ResNet(int(n_layers), **kwargs)
    else:
        raise NotImplementedError
