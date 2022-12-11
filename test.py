# import configs
import sys
import os
from cv2 import imshow
from matplotlib.image import imsave

import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as T

import matplotlib.pyplot as plt
from skimage.io import imsave

from config_utils.evaluate_args import obtain_evaluate_args
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from mypath import Path

import warnings
from tqdm import tqdm

from modeling.FCN import FCN8s
from modeling.unet import UNet
from modeling.pspnet import PSPNet
from modeling.deeplabv3 import Deeplabv3
from modeling.deeplabv3plus import Deeplabv3plus


def main():
    warnings.filterwarnings('ignore')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = True

    args = obtain_evaluate_args()
    model_path = args.model_path
    model_dir = os.path.dirname(model_path)
    save_dir = os.path.join(model_dir, 'Result')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    trans= T.Compose([
        # T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    images_base = os.path.join(Path.db_root_dir('loveda'), 'test', 'images_png')
    files = os.listdir(images_base)

    if args.dataset == 'loveda':
        args.num_classes = 7
    elif args.dataset == 'cityscapes':
        args.num_classes = 19
    else:
        pass
    
    if args.backbone == 'autodeeplab':
        net = Retrain_Autodeeplab(args)
    elif args.backbone == 'FCN':
        net = FCN8s(args.num_classes)
    elif args.backbone == 'unet':
        net = UNet(n_channels=3, n_classes=args.num_classes)
    elif args.backbone == 'pspnet':
        net = PSPNet(classes=args.num_classes)
    elif args.backbone == 'deeplabv3':
        net = Deeplabv3(classes=args.num_classes)
    elif args.backbone == 'deeplabv3+':
        net =  Deeplabv3plus(encoder_name='resnet50', encoder_weights=None, classes=args.num_classes)
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))
    

    if device == 'cpu':
        checkpoint = torch.load(args.model_path, map_location='cpu')
    else:
        checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device)
    net.eval()

    tbar = tqdm(files)
    for idx, name in enumerate(tbar):
        path = os.path.join(images_base, name)
        image = Image.open(path).convert('RGB')
        image = trans(image)
        image = torch.unsqueeze(image, dim=0).float()

        with torch.no_grad():
            image = image.to(device)
            if args.backbone == 'deeplabv3':
                output = net(image)['out']
            else:
                output = net(image)
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1).squeeze().astype(np.uint8)
            save_name = os.path.join(save_dir, name)
            imsave(save_name, pred)
            

if __name__ == '__main__':
    main()