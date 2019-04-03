#!/bin/bash

# I'm super lazy y'all

./main.sh --gin_config example_configs/biggan_posenc_imagenet32.gin \
	 --model_dir gs://octavian-training2/compare_gan/model/imagenet32/posenc \
	 --schedule continuous_eval \
	 $@