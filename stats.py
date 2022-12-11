from config_utils.evaluate_args import obtain_evaluate_args
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from modeling.FCN import FCN8s
from modeling.unet import UNet
from modeling.pspnet import PSPNet
from modeling.deeplabv3 import Deeplabv3
from modeling.deeplabv3plus import Deeplabv3plus

from torchstat import stat
from thop import profile
from thop import clever_format

import torch
import torch.backends.cudnn as cudnn

import warnings


def main():
    warnings.filterwarnings('ignore')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = True

    args = obtain_evaluate_args()

    if args.dataset == 'loveda':
        args.num_classes = 7
    elif args.dataset == 'cityscapes':
        args.num_classes = 19
    else:
        pass
    
    if args.backbone == 'autodeeplab':
        model = Retrain_Autodeeplab(args)
    elif args.backbone == 'FCN':
        model = FCN8s(args.num_classes)
    elif args.backbone == 'unet':
        model = UNet(n_channels=3, n_classes=args.num_classes)
    elif args.backbone == 'pspnet':
        model = PSPNet(classes=args.num_classes)
    elif args.backbone == 'deeplabv3':
        model = Deeplabv3(classes=args.num_classes)
    elif args.backbone == 'deeplabv3+':
        model =  Deeplabv3plus(encoder_name='resnet50', encoder_weights=None, classes=args.num_classes)
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

    stat(model, (3,1024,1024))


    input = torch.randn(1, 3, 1024, 1024)
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops:', flops)
    print('params:', params)
    print()

    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter: %.2fM' % (total/1e6))

if __name__ == '__main__':
    main()