CUDA_VISIBLE_DEVICES=0 python test.py \
--dataset loveda --backbone FCN \
--model_path retrain/loveda/deeplab-autodeeplab/model_best.pth.tar \
--net_arch search/loveda/loveda_exp/network_path_space.npy \
--cell_arch search/loveda/loveda_exp/genotype.npy \
--net_path search/loveda/loveda_exp/network_path.npy \
--filter_multiplier 10 --num_layers 8 \