import torch.nn as nn
import cell_level_search
from genotypes import PRIMITIVES
import torch.nn.functional as F
from operations import *
from decoding_formulas import Decoder

import sys

class AutoDeeplab(nn.Module):
    def __init__(self, num_classes, num_layers, criterion=None, filter_multiplier=8, block_multiplier=5, step=5, cell=cell_level_search.Cell):
        super(AutoDeeplab, self).__init__()

        self.cells = nn.ModuleList()
        self._num_layers = num_layers
        self._num_classes = num_classes
        self._step = step
        self._block_multiplier = block_multiplier
        self._filter_multiplier = filter_multiplier
        self._criterion = criterion
        
        #add alpha and beta to model's parameters
        self._initialize_alphas_betas()

        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial / 2)

        #as paper says, out_channels = B*F*S/4
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, half_f_initial * self._block_multiplier, 3, stride=1, padding=1),
            nn.BatchNorm2d(half_f_initial * self._block_multiplier),
            nn.ReLU()
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(half_f_initial * self._block_multiplier, half_f_initial * self._block_multiplier, 3, stride=2, padding=1),
            nn.BatchNorm2d(half_f_initial * self._block_multiplier),
            nn.ReLU()
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(half_f_initial * self._block_multiplier, f_initial * self._block_multiplier, 3, stride=2, padding=1),
            nn.BatchNorm2d(f_initial * self._block_multiplier),
            nn.ReLU()
        )

        # intitial_fm = C_initial
        for i in range(self._num_layers):

            if i == 0:
                cell1 = cell(self._step, self._block_multiplier, -1,
                             None, f_initial, None,
                             self._filter_multiplier)
                             
                cell2 = cell(self._step, self._block_multiplier, -1,
                             f_initial, None, None,
                             self._filter_multiplier * 2)
                self.cells += [cell1]
                self.cells += [cell2]
                
            elif i == 1:
                cell1 = cell(self._step, self._block_multiplier, f_initial,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier, self._filter_multiplier * 2, None,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 2, None, None,
                             self._filter_multiplier * 4)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]

            elif i == 2:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, None,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 4, None, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            elif i == 3:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            else:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, self._filter_multiplier * 8,
                             self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

        self.aspp_4 = nn.Sequential(
            ASPP(self._filter_multiplier * self._block_multiplier, self._num_classes, 24, 24)  # 96 / 4 as in the paper
        )
        self.aspp_8 = nn.Sequential(
            ASPP(self._filter_multiplier * 2 * self._block_multiplier, self._num_classes, 12, 12)  # 96 / 8
        )
        self.aspp_16 = nn.Sequential(
            ASPP(self._filter_multiplier * 4 * self._block_multiplier, self._num_classes, 6, 6)  # 96 / 16
        )
        self.aspp_32 = nn.Sequential(
            ASPP(self._filter_multiplier * 8 * self._block_multiplier, self._num_classes, 3, 3)  # 96 / 32
        )

    def forward(self, x):
        self.level_4 = []
        self.level_8 = []
        self.level_16 = []
        self.level_32 = []
        temp = self.stem0(x)
        temp = self.stem1(temp)
        self.level_4.append(self.stem2(temp))

        # Softmax on alphas and betas
        if torch.cuda.device_count() >= 1:
            # print('normalized betas on cuda(s)')
            img_device = torch.device('cuda', x.get_device())
            normalized_alphas = F.softmax(self.alphas, dim=-1)
            normalized_betas = torch.randn(self._num_layers, 4, 3).to(img_device)

            # normalized_betas[layer][ith node][2 : ➚, 1: ➙, 0 : ➘]
            for layer in range(len(self.betas)):
                if layer == 0:
                    normalized_betas[layer][0][1] = F.softmax(self.betas[layer][0][1], dim=-1)
                    normalized_betas[layer][1][0] = F.softmax(self.betas[layer][1][0], dim=-1)

                elif layer == 1:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1)
                    normalized_betas[layer][1][:2] = F.softmax(self.betas[layer][1][:2], dim=-1)
                    normalized_betas[layer][2][0] = F.softmax(self.betas[layer][2][0], dim=-1)

                elif layer == 2:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)
                    normalized_betas[layer][2][:2] = F.softmax(self.betas[layer][2][:2], dim=-1)
                    normalized_betas[layer][3][0] = F.softmax(self.betas[layer][3][0], dim=-1)

                else:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2], dim=-1)
                    normalized_betas[layer][3][:2] = F.softmax(self.betas[layer][3][:2], dim=-1)

        else:
            # print('normalized betas on cpu')
            normalized_alphas = F.softmax(self.alphas, dim=-1)
            normalized_betas = torch.randn(self._num_layers, 4, 3)


            for layer in range(len(self.betas)):
                if layer == 0:
                    normalized_betas[layer][0][1] = F.softmax(self.betas[layer][0][1], dim=-1)
                    normalized_betas[layer][1][0] = F.softmax(self.betas[layer][1][0], dim=-1)
                    # print(normalized_betas.device) cpu

                elif layer == 1:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1)
                    normalized_betas[layer][1][:2] = F.softmax(self.betas[layer][1][:2], dim=-1)
                    normalized_betas[layer][2][0] = F.softmax(self.betas[layer][2][0], dim=-1)

                elif layer == 2:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)
                    normalized_betas[layer][2][:2] = F.softmax(self.betas[layer][2][:2], dim=-1)
                    normalized_betas[layer][3][0] = F.softmax(self.betas[layer][3][0], dim=-1)
                else:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2], dim=-1)
                    normalized_betas[layer][3][:2] = F.softmax(self.betas[layer][3][:2], dim=-1)


        count = 0

        for layer in range(self._num_layers):

            if layer == 0:

                level4_new, = self.cells[count](None, None, self.level_4[-1], None, normalized_alphas)
                count += 1
                level8_new, = self.cells[count](None, self.level_4[-1], None, None, normalized_alphas)
                count += 1

                level4_new = normalized_betas[layer][0][1] * level4_new
                level8_new = normalized_betas[layer][1][0] * level8_new
                self.level_4.append(level4_new)
                self.level_8.append(level8_new)

            elif layer == 1:
                # print(self.level_4[-2].shape)
                # print(self.level_4[-1].shape)
                # print(self.level_8[-1].shape)
                # print(normalized_betas[layer][0])
                # sys.exit()
                # torch.Size([2, 40, 81, 81])
                # torch.Size([2, 40, 81, 81])
                # torch.Size([2, 80, 41, 41])
                # tensor([1.0904, 0.4998, 0.5002])

                level4_new_1, level4_new_2 = self.cells[count](self.level_4[-2],
                                                               None,
                                                               self.level_4[-1],
                                                               self.level_8[-1],
                                                               normalized_alphas)
                count += 1
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][0][2] * level4_new_2


                level8_new_1, level8_new_2 = self.cells[count](None,
                                                               self.level_4[-1],
                                                               self.level_8[-1],
                                                               None,
                                                               normalized_alphas)
                count += 1
                level8_new = normalized_betas[layer][1][0] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2

                level16_new, = self.cells[count](None,
                                                 self.level_8[-1],
                                                 None,
                                                 None,
                                                 normalized_alphas)
                level16_new = normalized_betas[layer][2][0] * level16_new
                count += 1

                self.level_4.append(level4_new)
                self.level_8.append(level8_new)
                self.level_16.append(level16_new)

            elif layer == 2:
                level4_new_1, level4_new_2 = self.cells[count](self.level_4[-2],
                                                               None,
                                                               self.level_4[-1],
                                                               self.level_8[-1],
                                                               normalized_alphas)
                count += 1
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][0][2] * level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count](self.level_8[-2],
                                                                             self.level_4[-1],
                                                                             self.level_8[-1],
                                                                             self.level_16[-1],
                                                                             normalized_alphas)
                count += 1
                level8_new = normalized_betas[layer][1][0] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][1][2] * level8_new_3

                level16_new_1, level16_new_2 = self.cells[count](None,
                                                                 self.level_8[-1],
                                                                 self.level_16[-1],
                                                                 None,
                                                                 normalized_alphas)
                count += 1
                level16_new = normalized_betas[layer][2][0] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2

                level32_new, = self.cells[count](None,
                                                 self.level_16[-1],
                                                 None,
                                                 None,
                                                 normalized_alphas)
                level32_new = normalized_betas[layer][3][0] * level32_new
                count += 1

                self.level_4.append(level4_new)
                self.level_8.append(level8_new)
                self.level_16.append(level16_new)
                self.level_32.append(level32_new)

            elif layer == 3:
                level4_new_1, level4_new_2 = self.cells[count](self.level_4[-2],
                                                               None,
                                                               self.level_4[-1],
                                                               self.level_8[-1],
                                                               normalized_alphas)
                count += 1
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][0][2] * level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count](self.level_8[-2],
                                                                             self.level_4[-1],
                                                                             self.level_8[-1],
                                                                             self.level_16[-1],
                                                                             normalized_alphas)
                count += 1
                level8_new = normalized_betas[layer][1][0] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][1][2] * level8_new_3

                # print(self.level_16[-2].size())
                # print(self.level_8[-1].size())
                # print(self.level_16[-1].size())
                # print(self.level_32[-1].size())
                # sys.exit()
                level16_new_1, level16_new_2, level16_new_3 = self.cells[count](self.level_16[-2],
                                                                                self.level_8[-1],
                                                                                self.level_16[-1],
                                                                                self.level_32[-1],
                                                                                normalized_alphas)
                count += 1
                level16_new = normalized_betas[layer][2][0] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][2][2] * level16_new_3

                level32_new_1, level32_new_2 = self.cells[count](None,
                                                                 self.level_16[-1],
                                                                 self.level_32[-1],
                                                                 None,
                                                                 normalized_alphas)
                count += 1
                level32_new = normalized_betas[layer][3][0] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2

                self.level_4.append(level4_new)
                self.level_8.append(level8_new)
                self.level_16.append(level16_new)
                self.level_32.append(level32_new)

            else:
                level4_new_1, level4_new_2 = self.cells[count](self.level_4[-2],
                                                               None,
                                                               self.level_4[-1],
                                                               self.level_8[-1],
                                                               normalized_alphas)
                count += 1
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][0][2] * level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count](self.level_8[-2],
                                                                             self.level_4[-1],
                                                                             self.level_8[-1],
                                                                             self.level_16[-1],
                                                                             normalized_alphas)
                count += 1

                level8_new = normalized_betas[layer][1][0] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][1][2] * level8_new_3

                level16_new_1, level16_new_2, level16_new_3 = self.cells[count](self.level_16[-2],
                                                                                self.level_8[-1],
                                                                                self.level_16[-1],
                                                                                self.level_32[-1],
                                                                                normalized_alphas)
                count += 1
                level16_new = normalized_betas[layer][2][0] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][2][2] * level16_new_3

                level32_new_1, level32_new_2 = self.cells[count](self.level_32[-2],
                                                                 self.level_16[-1],
                                                                 self.level_32[-1],
                                                                 None,
                                                                 normalized_alphas)
                count += 1
                level32_new = normalized_betas[layer][3][0] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2

                self.level_4.append(level4_new)
                self.level_8.append(level8_new)
                self.level_16.append(level16_new)
                self.level_32.append(level32_new)

            self.level_4 = self.level_4[-2:]
            self.level_8 = self.level_8[-2:]
            self.level_16 = self.level_16[-2:]
            self.level_32 = self.level_32[-2:]

        aspp_result_4 = self.aspp_4(self.level_4[-1])
        aspp_result_8 = self.aspp_8(self.level_8[-1])
        aspp_result_16 = self.aspp_16(self.level_16[-1])
        aspp_result_32 = self.aspp_32(self.level_32[-1])
        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)
        aspp_result_4 = upsample(aspp_result_4)
        aspp_result_8 = upsample(aspp_result_8)
        aspp_result_16 = upsample(aspp_result_16)
        aspp_result_32 = upsample(aspp_result_32)

        sum_feature_map = aspp_result_4 + aspp_result_8 + aspp_result_16 + aspp_result_32

        return sum_feature_map

    def _initialize_alphas_betas(self):
        #step is the number of block in a cell
        #k = 2 + 3 + 4 + ... + (step+1), it means all posible connections
        k = sum(1 for i in range(self._step) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        # alphas = torch.tensor(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)

        #some strange operations
        self.alphas = (1e-3 * torch.randn(k, num_ops)).clone().detach().requires_grad_(True)
        #numlayer * hidden units in layer * at most 3 uints from previous layer
        self.betas = (1e-3 * torch.randn(self._num_layers, 4, 3)).clone().detach().requires_grad_(True)

        self.alphas = nn.parameter.Parameter(self.alphas)
        self.betas = nn.parameter.Parameter(self.betas)

        self._arch_parameters = [
            self.alphas,
            self.betas,
        ]
        self._arch_param_names = [
            'alphas',
            'betas',
        ]
        
        for name, param in zip(self._arch_param_names, self._arch_parameters):
            self.register_parameter(name, param)
        
        # for name, param in self.named_parameters():
        #     print(name, param.size())

    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)


def main():
    model = AutoDeeplab(7, 8, None).cuda()
    x = torch.ones(2, 3, 321, 321).cuda()
    print(len(model.arch_parameters()))
    print(model.arch_parameters()[0].size())
    print(model.arch_parameters()[1].size())
    print(len(model.weight_parameters()))
    print(model(x).size())


if __name__ == '__main__':
    main()
