[TOC]

# Train CIFAR10 with Tensorflow

## Single GPU

### Experimental setup

- batch size: `128`
- optimizer: momentum with momentum set to `0.9`
- weight decay: `5e-4`
- learning rate
  - <50 epoch: `1e-1`
  - 50~100 epoch: `1e-2`
  - 100~150 epoch: `1e-3`
- dataset: CIFAR10
  - training data: *train* and *val* subset, 50000 images in total
  - testing data: *test* subset, 10000 images in total

### Accuracy

| Model | Accuracy, w/o EMA | Accuracy, w EMA |
| ----- | ----------------- | --------------- |
| VGG13 | 93.14%            |                 |
| VGG16 | 93.10%            |                 |

## Multi GPUs

