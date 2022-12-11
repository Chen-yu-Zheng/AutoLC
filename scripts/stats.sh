CUDA_VISIBLE_DEVICES=0 python stats.py \
--dataset loveda --backbone FCN \
--net_arch search/loveda/loveda_exp/network_path_space.npy \
--cell_arch search/loveda/loveda_exp/genotype.npy \
--net_path search/loveda/loveda_exp/network_path.npy \
--filter_multiplier 10 --num_layers 8 \