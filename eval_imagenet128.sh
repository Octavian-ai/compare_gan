nohup ./main.sh \
	--gin_config example_configs/biggan_imagenet128.gin \
    --model_dir gs://octavian-training2/compare_gan/model/imagenet128 \
	--schedule eval_last \
	--eval_batch_size 64 \
	--force_label 429 \
	--example_dir ./samples128 \
	--nouse_tpu $@ &