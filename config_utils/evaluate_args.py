import argparse


def obtain_evaluate_args():
    parser = argparse.ArgumentParser(description="Test autodeeplab on test set")
    parser.add_argument('--model_path', type=str, default="retrain/loveda/deeplab-autodeeplab/model_best.pth.tar", help='name of experiment')
    
    parser.add_argument('--dataset', type=str, default='loveda', help='pascal or cityscapes')
    parser.add_argument('--use_ABN', default=False, type=bool, help='whether use ABN')
    parser.add_argument('--dist', type=bool, default=False)
    parser.add_argument('--affine', default=False, type=bool, help='whether use affine in BN')

    parser.add_argument('--backbone', type=str, default='autodeeplab', help='backbone')
    parser.add_argument('--num_layers', default=10, type=int)
    parser.add_argument('--filter_multiplier', type=int, default=10)
    parser.add_argument('--block_multiplier', type=int, default=5)
    parser.add_argument('--net_arch', default='search/loveda/loveda_exp/network_path_space.npy', type=str)
    parser.add_argument('--cell_arch', default='search/loveda/loveda_exp/genotype.npy', type=str)
    parser.add_argument('--net_path', default='search/loveda/loveda_exp/network_path.npy', type=str)
    parser.add_argument('--dim', type=int, default=64, help='dim of AsDecoder and FPN')


    args = parser.parse_args()
    return args
