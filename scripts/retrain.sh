CUDA_VISIBLE_DEVICES=0 python retrain.py \
--dataset loveda --gpu_ids 0 \
--epochs 300 --base_lr 0.01 \
--net_arch search/loveda/loveda_exp/network_path_space.npy \
--cell_arch search/loveda/loveda_exp/genotype.npy \
--net_path search/loveda/loveda_exp/network_path.npy \
--crop_size 521 --batch_size 14 --filter_multiplier 20 \
--num_layers 10 \