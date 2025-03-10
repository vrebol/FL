from nets.QuantizedNets.utils.utils import filter_table, filter_unit_max, get_units, filter_table_unit
from nets.QuantizedNets.MobileNet.backward import QBWBlockGroupNorm
from nets.QuantizedNets.MobileNet.forward import QFWBlockGroupNorm
from nets.QuantizedNets.MobileNet.training import BlockGroupNorm
import random
import torch.nn as nn
import torch.nn.functional as F

from nets.Baseline.MobileNet.mobilenet_norm import IOLayer

import json

with open('nets/QuantizedNets/MobileNet/tables/table__CoCoFL_x64_QMobileNetGroupNorm.json', 'r') as fd:
    _g_table_qmobilenet = json.load(fd)
with open('nets/QuantizedNets/MobileNet/tables/table__Unit_x64_MobileNetGroupNorm.json', 'r') as fd:
    _g_table_qmobilenet_unit = json.load(fd)


class QMobileNetBase(nn.Module):
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, trained_block, fw_block, bw_block, num_classes, freeze_idxs=[]):
        super(QMobileNetBase, self).__init__()

        layer_idxs = [i for i in range(1 + 17 + 1 + 1)]  # 1 input layer 17 blocks 1 output layer
        assert set(freeze_idxs) <= set(layer_idxs), "Invalid layer idxs"

        self._trained_idxs = [idx for idx in layer_idxs if idx not in freeze_idxs]
        assert set(self._trained_idxs) == set([i for i in range(max(self._trained_idxs) \
                                                                + 1 - len(self._trained_idxs), max(self._trained_idxs) + 1)]), \
                 "No continous block of trained layers"

        self._trained_block = trained_block
        self._fw_block = fw_block
        self._bw_block = bw_block

        self.layer_in = IOLayer(3, 32, 3, 1, 1)
        self.layer_out = IOLayer(320, 1280, 1, 1, 0)
        self.linear = nn.Linear(1280, num_classes)

        if 0 not in self._trained_idxs:
            for parameter in self.layer_in.parameters():
                parameter.requires_grad = False

        if layer_idxs[-1] not in self._trained_idxs:
            for parameter in self.linear.parameters():
                parameter.requires_grad = False

        if layer_idxs[-2] not in self._trained_idxs:
            for parameter in self.layer_out.parameters():
                parameter.requires_grad = False

        self._block_idx = 1

        self.layers = self._make_layers(in_planes=32)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks - 1)
            for st in strides:
                # case fw layer (frozen)
                if self._block_idx < min(self._trained_idxs):
                    transition = True if (self._block_idx + 1) == min(min(self._trained_idxs), 18) else False
                    layers.append(self._fw_block(in_planes, out_planes, expansion, st, is_transition=transition))

                # case bw+(fw) layer (also frozen)
                elif self._block_idx > max(self._trained_idxs):
                    layers.append(self._bw_block(in_planes, out_planes, expansion, st))

                # trained layer
                elif self._block_idx in self._trained_idxs:
                    layers.append(self._trained_block(in_planes, out_planes, expansion, st))
                else:
                    raise ValueError
                in_planes = out_planes
                self._block_idx += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer_in(x)
        x = self.layers(x)
        x = self.layer_out(x)
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x




class QMobileNetGroupNorm(QMobileNetBase):
    def __init__(self, num_classes=10, freeze=[]):
        super(QMobileNetGroupNorm, self).__init__(BlockGroupNorm, QFWBlockGroupNorm, QBWBlockGroupNorm, num_classes, freeze_idxs=freeze)

    @staticmethod
    def n_freezable_layers():
        return 1 + 17 + 1 + 1

    @staticmethod
    def get_freezing_config(resources):
        configs = filter_table(resources, _g_table_qmobilenet, QMobileNetGroupNorm.n_freezable_layers())
        return random.choice(configs)
    
    @staticmethod
    def get_max_resources(unit):
        # extract max from unit table 
        max_resources = filter_unit_max(unit, _g_table_qmobilenet_unit)
        return max_resources

    @staticmethod
    def get_freezing_configs_unit(unit):
        configs = filter_table_unit(unit, _g_table_qmobilenet_unit)
        return configs
    
    @staticmethod
    def get_units():
        units = get_units(_g_table_qmobilenet_unit)
        return units
    
    @staticmethod
    def get_min_req_resources():
        units = QMobileNetGroupNorm.get_units()
        min_resources = [1,1,1]
        for unit in units:
            max_unit_resources = QMobileNetGroupNorm.get_max_resources(unit)
            for i in range(3):
                if max_unit_resources[i] < min_resources[i]:
                    min_resources[i] = max_unit_resources[i]
        return min_resources