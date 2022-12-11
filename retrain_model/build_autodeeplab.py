import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from operations import FPN, AssymetricDecoder, NaiveBN, ABN
from retrain_model.aspp import ASPP
from retrain_model.decoder import Decoder
from retrain_model.new_model import get_default_arch, newModel

from torch.nn import functional as F


class Retrain_Autodeeplab(nn.Module):
    def __init__(self, args):
        super(Retrain_Autodeeplab, self).__init__()
        self.args = args
        self.filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}

        BatchNorm2d = ABN if args.use_ABN else NaiveBN
        if (not args.dist and args.use_ABN) or (args.dist and args.use_ABN and dist.get_rank() == 0):
            print("=> use ABN!")
        
        if args.net_arch is not None and args.cell_arch is not None:
            network_arch, cell_arch, network_path = np.load(args.net_arch), np.load(args.cell_arch), np.load(args.net_path)
        else:
            network_arch, cell_arch, network_path = get_default_arch()
            
        self.encoder = newModel(network_arch, cell_arch, args.num_classes, args.num_layers, args.filter_multiplier, BatchNorm=BatchNorm2d, args=args)
        
        self.aspp = ASPP(args.filter_multiplier * args.block_multiplier * self.filter_param_dict[network_path[-1]],
                         256, args.num_classes, conv=nn.Conv2d, norm=BatchNorm2d)
        
        self.decoder = Decoder(args.num_classes, filter_multiplier=args.filter_multiplier * args.block_multiplier,
                               args=args, last_level=network_path[-1])
        
        self.mmax = network_path.max()

        if self.mmax == 3:
            self.fpn = FPN(
                in_channels_list=[
                    self.args.filter_multiplier * self.args.block_multiplier * self.filter_param_dict[0],
                    self.args.filter_multiplier * self.args.block_multiplier * self.filter_param_dict[1],
                    self.args.filter_multiplier * self.args.block_multiplier * self.filter_param_dict[2],
                    self.args.filter_multiplier * self.args.block_multiplier * self.filter_param_dict[3]
                ],
                out_channels= args.dim
            )
            self.asdecoder = AssymetricDecoder(
                in_channels= args.dim,
                out_channels= self.args.filter_multiplier * self.args.block_multiplier * self.filter_param_dict[0],
                in_feat_output_strides=(4,8,16,32),
                out_feat_output_stride=4
            )
        
        elif self.mmax == 2:
            self.fpn = FPN(
                in_channels_list=[
                    self.args.filter_multiplier * self.args.block_multiplier * self.filter_param_dict[0],
                    self.args.filter_multiplier * self.args.block_multiplier * self.filter_param_dict[1],
                    self.args.filter_multiplier * self.args.block_multiplier * self.filter_param_dict[2],
                    0
                ],
                out_channels= args.dim
            )
            self.asdecoder = AssymetricDecoder(
                in_channels= args.dim,
                out_channels= self.args.filter_multiplier * self.args.block_multiplier * self.filter_param_dict[0],
                in_feat_output_strides=(4,8,16),
                out_feat_output_stride=4
            )
        
        elif self.mmax == 1:
            self.fpn = FPN(
                in_channels_list=[
                    self.args.filter_multiplier * self.args.block_multiplier * self.filter_param_dict[0],
                    self.args.filter_multiplier * self.args.block_multiplier * self.filter_param_dict[1],
                    0,
                    0
                ],
                out_channels= args.dim
            )
            self.asdecoder = AssymetricDecoder(
                in_channels= args.dim,
                out_channels= self.args.filter_multiplier * self.args.block_multiplier * self.filter_param_dict[0],
                in_feat_output_strides=(4,8),
                out_feat_output_stride=4
            )
        
        else:
            self.fpn = FPN(
                in_channels_list=[
                    self.args.filter_multiplier * self.args.block_multiplier * self.filter_param_dict[0],
                    0,
                    0,
                    0
                ],
                out_channels= args.dim
            )
            self.asdecoder = AssymetricDecoder(
                in_channels= args.dim,
                out_channels= self.args.filter_multiplier * self.args.block_multiplier * self.filter_param_dict[0],
                in_feat_output_strides=(4),
                out_feat_output_stride=4
            )

    def forward(self, x):
        encoder_output, os4_feature, os8_feature, os16_feature, os32_feature = self.encoder(x)
        high_level_feature = self.aspp(encoder_output)

        if self.mmax == 3:
            oss = [os4_feature, os8_feature, os16_feature, os32_feature]
            oss = self.fpn(oss)
            os4_feature = self.asdecoder(oss)
        
        elif self.mmax == 2:
            oss = [os4_feature, os8_feature, os16_feature]
            oss = self.fpn(oss)
            os4_feature = self.asdecoder(oss)
        
        elif self.mmax == 1:
            oss = [os4_feature, os8_feature]
            oss = self.fpn(oss)
            os4_feature = self.asdecoder(oss)
        
        else:
            oss = [os4_feature]
            oss = self.fpn(oss)
            os4_feature = self.asdecoder(oss)

        decoder_output = self.decoder(high_level_feature, os4_feature)

        return nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)(decoder_output)

    def get_params(self):
        back_bn_params, back_no_bn_params = self.encoder.get_params()
        tune_wd_params = list(self.aspp.parameters()) \
                         + list(self.decoder.parameters()) \
                         + back_no_bn_params
        no_tune_wd_params = back_bn_params
        return tune_wd_params, no_tune_wd_params