import os

import tensorflow as tf

HEIGHT = 32
WIDTH = 32
DEPTH = 3


class Cifar10DataSet(object):
    """Cifar10 data set.
    Described by http://www.cs.toronto.edu/~kriz/cifar.html.
    """

    def __init__(self, data_dir, subset='trainval', use_distortion=True):
        self.data_dir = data_dir
        self.subset = subset
        self.use_distortion = use_distortion

    def get_filenames(self):
        if self.subset in ['trainval', 'test']:
            return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
        else:
            raise ValueError('Invalid data subset "%s"' % self.subset)

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([DEPTH * HEIGHT * WIDTH])

        # Reshape from [depth * height * width] to [depth, height, width].
        image = tf.cast(
            tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
            tf.float32)
        label = tf.cast(features['label'], tf.int32)

        # Custom preprocessing.
        image = self.preprocess(image)

        return image, label

    def make_batch(self, batch_size, n_epochs=1):
        """Read the images and labels from 'filenames'."""
        filenames = self.get_filenames()
        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset(filenames).repeat(n_epochs)

        # Parse records.
        dataset = dataset.map(
            self.parser, num_parallel_calls=batch_size)

        # Potentially shuffle records.
        if self.subset == 'trainval':
            min_queue_examples = int(
                Cifar10DataSet.num_examples_per_epoch(self.subset) * 0.4)
            # Ensure that the capacity is sufficiently large to provide good random
            # shuffling.
            dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

        # Batch it up.
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()

        return image_batch, label_batch

    def preprocess(self, image):
        """Preprocess a single image in [height, width, depth] layout."""
        if self.subset == 'trainval' and self.use_distortion:
            # Pad 4 pixels on each dimension of feature map, done in mini-batch
            image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
            image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
            image = tf.image.random_flip_left_right(image)
        image = (image - 125.) / 255.
        return image

    @staticmethod
    def num_examples_per_epoch(subset='trainval'):
        if subset == 'trainval':
            return 50000
        elif subset == 'test':
            return 10000
        else:
            raise ValueError('Invalid data subset "%s"' % subset)


def multi_gpu_input_fn(data_dir, subset, n_towers, batch_size, n_epochs=1):
    batch_size_per_device = batch_size // n_towers
    tower_images, tower_labels = [], []
    for i in range(n_towers):
        with tf.device('/cpu:%d' % i):
            with tf.name_scope('InputFunction-%d' % i):
                data = Cifar10DataSet(data_dir, subset, subset == 'trainval')
                image_batch, label_batch = data.make_batch(batch_size_per_device, n_epochs)
                if n_towers == 1:
                    return [image_batch], [label_batch]
                tower_images.append(image_batch)
                tower_labels.append(label_batch)
    return tower_images, tower_labels
