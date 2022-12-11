CUDA_VISIBLE_DEVICES=3 python search.py \
 --batch_size 2 --dataset loveda --checkname loveda_exp \
 --alpha_epoch 30 --filter_multiplier 8 --num_layers 10\
 --resize 512 --crop_size 321 \
 --gpu_ids 0 --epochs 60 --use_amp