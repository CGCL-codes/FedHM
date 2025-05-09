"""Ref to HeteroFL pre-activated ResNet18"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
# from ..bn_ops import get_bn_layer
from .slimmable_nets import BaseModule, SlimmableMixin
from .slimmable_ops import SlimmableConv2d, SlimmableBatchNorm2d, SlimmableLinear


hidden_size = [64, 128, 256, 512]


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, norm_layer, conv_layer, fix_out=False,
                 fix_in=False):
        super(Block, self).__init__()
        # self.norm_layer = norm_layer
        if fix_in:
            self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                                    bias=False, non_slimmable_in=True)
            self.n1 = norm_layer(planes, non_slimmable=True)
        else:
            self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                                    bias=False)
            self.n1 = norm_layer(planes)

        if fix_out:
            self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
                                    non_slimmable_out=fix_out)
        else:
            self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.n2 = norm_layer(planes)

        non_slimmable_in = fix_out or fix_in
        if stride != 1 or in_planes != self.expansion * planes or non_slimmable_in:
            self.shortcut = nn.Sequential(
                conv_layer(in_planes, self.expansion * planes, kernel_size=1,
                           stride=stride, bias=False, non_slimmable_in = non_slimmable_in),
                norm_layer(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.n1(self.conv1(x)))
        out = F.relu(self.n2(self.conv2(out)))

        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x

        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, norm_layer, conv_layer, fix_out=False,
                 fix_in=False):
        super(Bottleneck, self).__init__()
        assert not fix_out
        assert not fix_in
        # self.norm_layer = norm_layer
        self.n1 = norm_layer(in_planes)
        self.conv1 = conv_layer(in_planes, planes, kernel_size=1, bias=False)
        self.n2 = norm_layer(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n3 = norm_layer(planes)
        self.conv3 = conv_layer(planes, self.expansion * planes, kernel_size=1, bias=False,
                                non_slimmable_out=fix_out)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = conv_layer(in_planes, self.expansion * planes, kernel_size=1,
                                       stride=stride, bias=False, non_slimmable_out=fix_out)

    def forward(self, x):
        out = F.relu(self.n1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.n2(out)))
        out = self.conv3(F.relu(self.n3(out)))
        out += shortcut
        return out


class ResNet(BaseModule, SlimmableMixin):
    input_shape = [None, 3, 32, 32]

    def __init__(self, hidden_size, block, num_blocks, num_classes=10, bn_type='bn',
                 track_running_stats=True, width_scale=1., share_affine=False, slimmabe_ratios=None):
        super(ResNet, self).__init__()
        self._set_slimmabe_ratios(slimmabe_ratios)

        if width_scale != 1.:
            hidden_size = [int(hs * width_scale) for hs in hidden_size]
        if bn_type.startswith('d'):
            print("WARNING: When using dual BN, you should not do slimming.")
        if track_running_stats:
            print("WARNING: We cannot track running_stats when slimmable BN is used.")
        self.bn_type = bn_type
        if bn_type == 'bn':
            norm_layer = lambda n_ch, **kwargs: SlimmableBatchNorm2d(
                n_ch, track_running_stats=track_running_stats, **kwargs)
        elif bn_type == 'dbn':
            from .bn_ops import DualNormLayer
            assert not share_affine, "We don't recommend to share affine."
            norm_layer = lambda n_ch: DualNormLayer(
                n_ch, track_running_stats=track_running_stats, affine=True,
                bn_class=SlimmableBatchNorm2d, share_affine=share_affine)
        else:
            raise RuntimeError(f"Not support bn_type={bn_type}")
        conv_layer = SlimmableConv2d

        self.in_planes = hidden_size[0]
        self.conv1 = SlimmableConv2d(3, hidden_size[0], kernel_size=3, stride=1, padding=1,
                                     bias=False, non_slimmable_in=True)
        self.n1 = norm_layer(hidden_size[0])

        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1,
                                       norm_layer=norm_layer, conv_layer=conv_layer)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2,
                                       norm_layer=norm_layer, conv_layer=conv_layer)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2,
                                       norm_layer=norm_layer, conv_layer=conv_layer)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2,
                                       norm_layer=norm_layer, conv_layer=conv_layer)

        self.linear = SlimmableLinear(hidden_size[3] * block.expansion, num_classes,
                                      non_slimmable_out=True)

    def _make_layer(self, block, planes, num_blocks, stride, norm_layer, conv_layer,
                    fix_out=False, fix_in=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i_layer, stride in enumerate(strides):
            layers.append(
                block(self.in_planes, planes, stride, norm_layer, conv_layer,
                      fix_out=False if (not fix_out) or (i_layer < num_blocks - 1) else fix_out,
                      fix_in=False if (not fix_in) or (i_layer > 0) else fix_in))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_pre_clf_fea=False):
        out = self.conv1(x)
        out = F.relu(self.n1(out))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        if return_pre_clf_fea:
            return logits, out
        else:
            return logits

    def print_footprint(self):
        input_shape = self.input_shape
        input_shape[0] = 2
        x = torch.rand(input_shape)
        batch = x.shape[0]
        print(f"input: {np.prod(x.shape[1:])} <= {x.shape[1:]}")
        x = self.conv1(x)
        print(f"conv1: {np.prod(x.shape[1:])} <= {x.shape[1:]}")
        for i_layer, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)
            print(f"layer {i_layer}: {np.prod(x.shape[1:]):5d} <= {x.shape[1:]}")

def init_param(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    return m


# Instantiations
def resnet18(**kwargs):
    model = ResNet(hidden_size, Block, [2, 2, 2, 2], **kwargs)
    model.apply(init_param)
    return model


def resnet26(**kwargs):
    model = ResNet(hidden_size, Block, [3, 3, 3, 3], **kwargs)
    model.apply(init_param)
    return model


def resnet34(**kwargs):
    model = ResNet(hidden_size, Block, [3, 4, 6, 3], **kwargs)
    model.apply(init_param)
    return model


def resnet50(**kwargs):
    model = ResNet(hidden_size, Bottleneck, [3, 4, 6, 3], **kwargs)
    model.apply(init_param)
    return model


def resnet101(**kwargs):
    model = ResNet(hidden_size, Bottleneck, [3, 4, 23, 3], **kwargs)
    model.apply(init_param)
    return model


def resnet152(**kwargs):
    model = ResNet(hidden_size, Bottleneck, [3, 8, 36, 3], **kwargs)
    model.apply(init_param)
    return model




