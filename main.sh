#!/bin/sh

# Written to run on ctpu shells

PYTHONPATH=. python3 compare_gan/main.py \
	--model_dir gs://octavian-training2/compare_gan/model/imagenet32/$RANDOM \
	--tfds_data_dir gs://octavian-training2/compare_gan/data/imagenet/ \
	$@