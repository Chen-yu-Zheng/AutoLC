CUDA_VISIBLE_DEVICES=3 python retrain.py \
 --dataset loveda --gpu_ids 0 --checkname numlayer10 \
 --epochs 100 --base_lr 0.005 --eval_interval 10 \
 --net_arch search/loveda/loveda_exp/search_experiment_5/network_path_space.npy \
 --cell_arch search/loveda/loveda_exp/search_experiment_5/genotype.npy \
 --net_path /search/loveda/loveda_exp/search_experiment_5/network_path.npy \
 --crop_size 769 --batch_size 8 --filter_multiplier 20 \
 --num_layers 10 \
 --resume retrain/loveda/deeplab-autodeeplab/model_best.pth.tar \
 --ft \