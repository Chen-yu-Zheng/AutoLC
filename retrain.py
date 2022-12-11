import os
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from collections import OrderedDict
from mypath import Path #get dataset root
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
# from modeling.deeplab import *
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args
from utils.copy_state_dict import copy_state_dict

import pdb
import warnings

import torch
import torch.utils.data
import torch.backends.cudnn
import torch.optim as optim

from utils.utils import AverageMeter
from utils.loss import build_criterion
from utils.step_lr_scheduler import Iter_LR_Scheduler
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args

from modeling.FCN import FCN8s
from modeling.unet import UNet
from modeling.pspnet import PSPNet
from modeling.deeplabv3 import Deeplabv3
from modeling.deeplabv3plus import Deeplabv3plus

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


print('working with pytorch version {}'.format(torch.__version__))
print('with cuda version {}'.format(torch.version.cuda))
print('cudnn enabled: {}'.format(torch.backends.cudnn.enabled))
print('cudnn version: {}'.format(torch.backends.cudnn.version()))


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(self.args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.use_amp = True if (APEX_AVAILABLE and self.args.use_amp) else False
        self.opt_level = self.args.opt_level

        kwargs = {'num_workers': self.args.workers, 'pin_memory': True, 'drop_last':True}

        # cityscapes
        # self.train_loaderA, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        
        # loveda
        self.train_loader, self.val_loader, self.args.num_classes = make_data_loader(self.args, **kwargs)

        if self.args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(self.args.dataset), self.args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                self.args.weight = np.load(classes_weights_path)
            else:
                raise NotImplementedError
                #if so, which trainloader to use?
                # weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            self.args.weight = torch.from_numpy(self.args.weight.astype(np.float32))
        else:
            self.args.weight = None
        
        if self.args.criterion == 'Ohem':
            self.args.thresh = 0.7
            self.args.crop_size = [self.args.crop_size, self.args.crop_size] if isinstance(self.args.crop_size, int) else self.args.crop_size
            self.args.n_min = int((self.args.batch_size / len(self.args.gpu_ids) * self.args.crop_size[0] * self.args.crop_size[1]) // 16)

        self.criterion = build_criterion(self.args)

        # Define network
        if self.args.backbone == 'autodeeplab':
            model = Retrain_Autodeeplab(self.args)
        elif self.args.backbone == 'FCN':
            model = FCN8s(self.args.num_classes)
        elif self.args.backbone == 'unet':
            model = UNet(n_channels=3, n_classes=self.args.num_classes)
        elif self.args.backbone == 'pspnet':
            model = PSPNet(classes=self.args.num_classes)
        elif self.args.backbone == 'deeplabv3':
            model = Deeplabv3(classes=self.args.num_classes)
        elif self.args.backbone == 'deeplabv3+':
            model =  Deeplabv3plus(encoder_name='resnet50', encoder_weights=None, classes=self.args.num_classes)
        else:
            raise ValueError('Unknown backbone: {}'.format(self.args.backbone))
        
        if len(self.args.gpu_ids) > 1:
            optimizer = optim.SGD(model.module.parameters(), lr=self.args.base_lr, momentum=0.9, weight_decay=0.0001)
        else:
            optimizer = optim.SGD(model.parameters(), lr=self.args.base_lr, momentum=0.9, weight_decay=0.0001)

        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.args.num_classes)

        # Define lr scheduler
        max_iteration = len(self.train_loader) * self.args.epochs
        self.scheduler = Iter_LR_Scheduler(self.args, max_iteration, len(self.train_loader))

        # Using data parallel
        if self.args.cuda and len(self.args.gpu_ids) >1:
            if self.opt_level == 'O2' or self.opt_level == 'O3':
                print('currently cannot run with nn.DataParallel and optimization level', self.opt_level)
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            print('training on multiple-GPUs')
        
        elif self.args.cuda and len(self.args.gpu_ids) == 1:
            self.model = self.model.cuda()
        
        else:
            pass
        
        if self.args.freeze_bn:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False


        # Resuming checkpoint
        self.best_pred = 0.0

        # resume training
        if self.args.resume is not None:
            if not os.path.isfile(self.args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(self.args.resume))

            checkpoint = torch.load(self.args.resume)
            self.args.start_epoch = checkpoint['epoch']

            # if the weights are wrapped in module object we have to clean it
            if self.args.clean_module:
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                state_dict = checkpoint['state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove 'module.' of dataparallel
                    new_state_dict[name] = v
                # self.model.load_state_dict(new_state_dict)
                copy_state_dict(self.model.state_dict(), new_state_dict)

            else:
                if torch.cuda.device_count() > 1 or self.args.load_parallel:
                    self.model.module.load_state_dict(checkpoint['state_dict'])

                else:
                    self.model.load_state_dict(checkpoint['state_dict'])

            # finetuning on a different dataset(False)
            if not self.args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.best_pred = checkpoint['best_pred']

            print("=> loaded checkpoint '{}' (epoch {})".format(self.args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if self.args.ft:
            self.args.start_epoch = 0

    def training(self, epoch):

        train_loss = AverageMeter()
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            cur_iter = epoch * len(self.train_loader) + i
            self.scheduler(self.optimizer, cur_iter)

            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            
            if self.args.backbone == 'deeplabv3':
                output = self.model(image)['out']
            else:
                output = self.model(image)
            loss = self.criterion(output, target)

            if np.isnan(loss.item()) or np.isinf(loss.item()):
                pdb.set_trace()
            train_loss.update(loss.item(), self.args.batch_size)

            self.optimizer.zero_grad()

            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()

            tbar.set_description('Train iter Loss: %.3f, lr: %.6f' % (train_loss.val, self.scheduler.get_lr(self.optimizer)))
            self.writer.add_scalar('train/lr', self.scheduler.get_lr(self.optimizer), i + num_img_tr * epoch)
            self.writer.add_scalar('train/train_loss_step', train_loss.val, i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
        if (epoch + 1) % self.args.eval_interval == 0:
            global_step = epoch + 1
            self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/train_avg_loss_epoch', train_loss.avg, epoch + 1)
        print('[Epoch: %d, numImages: %5d]' % (epoch + 1, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss.avg)
        
        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            if len(self.args.gpu_ids) > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        val_loss = AverageMeter()

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                if self.args.backbone == 'deeplabv3':
                    output = self.model(image)['out']
                else:
                    output = self.model(image)
            loss = self.criterion(output, target)

            if np.isnan(loss.item()) or np.isinf(loss.item()):
                pdb.set_trace()
            val_loss.update(loss.item(), self.args.batch_size)

            tbar.set_description('Val iter loss: %.3f' % (val_loss.val))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/avg_loss_epoch', val_loss.avg, epoch + 1)
        self.writer.add_scalar('val/mIoU', mIoU, epoch + 1)
        self.writer.add_scalar('val/Acc', Acc, epoch + 1)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch + 1)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch + 1)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch + 1, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % val_loss.avg)

        new_pred = mIoU

        if new_pred >= self.best_pred:
            is_best = True
            self.best_pred = new_pred

            if len(self.args.gpu_ids) > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()

            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
        else:
            is_best = False
            if len(self.args.gpu_ids) > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

def main():
    warnings.filterwarnings('ignore')
    args = obtain_retrain_autodeeplab_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    #并行化的batchnorm
    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 3000,
            'cityscapes': 4000,
            'pascal': 5000,
            'kd':1000,
            'loveda':300
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 2 * len(args.gpu_ids)

    if args.checkname is None:
        args.checkname = args.backbone

    print(args)


    seed_torch(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
