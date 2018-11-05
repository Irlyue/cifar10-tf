export model_dir=/tmp/cifar10
export network=vgg-16


if [ "$1" == "train" ]; then
    export CUDA_VISIBLE_DEVICES='4'
    export n_devices=1
    export n_epochs=50
    export data=trainval

    if [ -e $model_dir ];then
        echo "Removing old checkpoint directory..."
        rm -rf $model_dir
    fi

	lr=1e-1 python train.py
	lr=1e-2 python train.py
	lr=1e-3 python train.py

elif [ "$1" == "test" ]; then
    export CUDA_VISIBLE_DEVICES='1'
    export n_devices=1
    export data=test

	python evaluate.py
fi
