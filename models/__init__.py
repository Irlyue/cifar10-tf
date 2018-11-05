from .vgg import VGG


def load_network(network):
    network = network.lower()
    if network.startswith('vgg'):
        _, n_layers = network.split('-')
        return lambda **kwargs: VGG(int(n_layers), **kwargs)
