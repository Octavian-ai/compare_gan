#!/bin/sh

PREFIX=gs://octavian-training2/compare_gan/imagenet32

python3 compare_gan/main.py \
	--model_dir $PREFIX/model/$RANDOM \
	--tfds_data_dir $PREFIX/data \
	$@