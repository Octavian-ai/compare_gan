#!/bin/bash

nohup ./main.sh \
	-gin_config example_configs/biggan_posenc_imagenet128.gin \
	-model_dir gs://octavian-training2/compare_gan/model/imagenet128_posenc &