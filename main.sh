#!/bin/sh

# 	--model_dir gs://octavian-training2/compare_gan/model/imagenet32/$RANDOM \

# Written to run on ctpu shells

PYTHONPATH=. python3 compare_gan/main.py \
	--tfds_data_dir gs://octavian-training2/compare_gan/data/imagenet/ \
	$@