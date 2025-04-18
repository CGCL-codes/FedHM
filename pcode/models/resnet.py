# -*- coding: utf-8 -*-
import math
import torch.nn as nn
import torch.nn.init as init
from torchvision.models import resnet
from pcode.models.lowrank_resnet import LowRankBasicBlockConv1x1,LowRankBottleneckConv1x1,HybridResNet,FullRankBasicBlock


__all__ = ["resnet","ResNet_imagenet"]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding."
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )

def norm2d(group_norm_num_groups, planes,track_running_stats = True):
    if group_norm_num_groups is not None and group_norm_num_groups > 0:
        # group_norm_num_groups == planes -> InstanceNorm
        # group_norm_num_groups == 1 -> LayerNorm
        group_nums = planes // group_norm_num_groups
        return nn.GroupNorm(group_nums, planes)
    else:
        return nn.BatchNorm2d(planes,track_running_stats = track_running_stats)

class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
        #self.register_buffer("rate",torch.tensor(rate))

    def forward(self, input):
        output = input / self.rate if self.training else input
        return output


class PruningBasicBlock(nn.Module):
    """
    [3 * 3, 64]
    [3 * 3, 64]
    """

    expansion = 1

    def __init__(
            self,
            in_planes,
            out_planes,
            stride=1,
            downsample=None,
            group_norm_num_groups=None,
            track_running_stats=True,
            rate = 1,
            need_scaler = False,
            global_rate = 1
    ):
        super(PruningBasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, int(round(out_planes)), stride)
        self.bn1 = norm2d(group_norm_num_groups, planes=int(round(out_planes)), track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(int(round(out_planes)), int(round(out_planes)))
        self.bn2 = norm2d(group_norm_num_groups, planes=int(round(out_planes)), track_running_stats=track_running_stats)

        self.downsample = downsample
        self.stride = stride

        self.need_scaler = need_scaler
        if self.need_scaler:
            self.scaler = Scaler(rate / global_rate)

        # print(rate)
        # print(rate / global_rate, self.need_scaler)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.need_scaler:
            out = self.scaler(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.need_scaler:
            out = self.scaler(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class PruningBottleneck(nn.Module):
    """
    [1 * 1, x]
    [3 * 3, x]
    [1 * 1, x * 4]
    """

    expansion = 4

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        group_norm_num_groups=None,
        track_running_stats = True,
        rate = 1,
        need_scaler = False,
        global_rate = 1
    ):
        super(PruningBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_planes, out_channels=int(round(out_planes)), kernel_size=1, bias=False
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=int(round(out_planes)),track_running_stats = track_running_stats)

        self.conv2 = nn.Conv2d(
            in_channels=int(round(out_planes)),
            out_channels=int(round(out_planes)),
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = norm2d(group_norm_num_groups, planes=int(round(out_planes)),track_running_stats = track_running_stats)

        self.conv3 = nn.Conv2d(
            in_channels= int(round(out_planes)),
            out_channels= int(round(out_planes * 4)),
            kernel_size=1,
            bias=False,
        )
        self.bn3 = norm2d(group_norm_num_groups, planes=int(round(out_planes * 4)), track_running_stats = track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

        self.need_scaler = need_scaler
        if self.need_scaler:
            self.scaler = Scaler(rate / global_rate)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.need_scaler:
            out = self.scaler(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.need_scaler:
            out = self.scaler(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.need_scaler:
            out = self.scaler(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    """
    [3 * 3, 64]
    [3 * 3, 64]
    """

    expansion = 1

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        group_norm_num_groups=None,
        track_running_stats = True,
        rate = 1,
        need_scaler = False,
        global_rate = 1
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, int(round(out_planes)), stride)
        self.bn1 = norm2d(group_norm_num_groups, planes=int(round(out_planes)), track_running_stats = track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(int(round(out_planes)), int(round(out_planes)))
        self.bn2 = norm2d(group_norm_num_groups, planes=int(round(out_planes)), track_running_stats = track_running_stats)

        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    [1 * 1, x]
    [3 * 3, x]
    [1 * 1, x * 4]
    """

    expansion = 4

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        group_norm_num_groups=None,
        track_running_stats = True,
        rate=1,
        need_scaler = False,
        rank_factor = 1,
        square = False
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_planes, out_channels=int(round(out_planes)), kernel_size=1, bias=False
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=int(round(out_planes)), track_running_stats=track_running_stats)

        self.conv2 = nn.Conv2d(
            in_channels=int(round(out_planes)),
            out_channels=int(round(out_planes)),
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = norm2d(group_norm_num_groups, planes=int(round(out_planes)), track_running_stats=track_running_stats)

        self.conv3 = nn.Conv2d(
            in_channels=int(round(out_planes)),
            out_channels=int(round(out_planes * 4)),
            kernel_size=1,
            bias=False,
        )
        self.bn3 = norm2d(group_norm_num_groups, planes=int(round(out_planes * 4)),
                          track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def decide_num_classes(dataset):
    if dataset == "cifar10" or dataset == "svhn":
        return 10
    elif dataset == "cifar100":
        return 100
    elif "tiny" in dataset:
        return 200
    elif "imagenet" in dataset:
        return 1000
    elif "femnist" == dataset:
        return 62

class ResNetBase(nn.Module):
    def _decide_num_classes(self):
        if self.dataset == "cifar10" or self.dataset == "svhn":
            return 10
        elif self.dataset == "cifar100":
            return 100
        elif "tiny" in self.dataset:
            return 200
        elif "imagenet" in self.dataset:
            return 1000
        elif "femnist" == self.dataset:
            return 62

    def _weight_initialization(self):
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(mean=0, std=0.01)
            #     m.bias.data.zero_()

    def _make_block(
        self, block_fn, planes, block_num, stride=1, group_norm_num_groups=None, track_running_stats = True
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block_fn.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm2d(group_norm_num_groups, planes=planes * block_fn.expansion,track_running_stats=track_running_stats),
            )

        layers = []
        layers.append(
            block_fn(
                in_planes=self.inplanes,
                out_planes=planes,
                stride=stride,
                downsample=downsample,
                group_norm_num_groups=group_norm_num_groups,
                track_running_stats = track_running_stats,

            )
        )
        self.inplanes = planes * block_fn.expansion

        for _ in range(1, block_num):
            layers.append(
                block_fn(
                    in_planes=self.inplanes,
                    out_planes=planes,
                    group_norm_num_groups=group_norm_num_groups,
                    track_running_stats=track_running_stats
                )
            )
        return nn.Sequential(*layers)

    def train(self, mode=True):
        super(ResNetBase, self).train(mode)

        # if self.freeze_bn:
        #     for m in self.modules():
        #         if isinstance(m, nn.BatchNorm2d):
        #             m.eval()
        #             if self.freeze_bn_affine:
        #                 m.weight.requires_grad = False
        #                 m.bias.requires_grad = False

def decide_model_params(pruning = False):
    if pruning == False:
        model_params = {
            18: {"block": BasicBlock, "layers": [2, 2, 2, 2]},
            34: {"block": BasicBlock, "layers": [3, 4, 6, 3]},
            50: {"block": Bottleneck, "layers": [3, 4, 6, 3]},
            101: {"block": Bottleneck, "layers": [3, 4, 23, 3]},
            152: {"block": Bottleneck, "layers": [3, 8, 36, 3]},
        }
    else:
        model_params = {
            18: {"block": PruningBasicBlock, "layers": [2, 2, 2, 2]},
            34: {"block": PruningBasicBlock, "layers": [3, 4, 6, 3]},
            50: {"block": PruningBottleneck, "layers": [3, 4, 6, 3]},
            101: {"block": PruningBottleneck, "layers": [3, 4, 23, 3]},
            152: {"block": PruningBottleneck, "layers": [3, 8, 36, 3]},
        }


    return model_params

class ResNet_imagenet(ResNetBase):
    def __init__(
        self,
        dataset,
        resnet_size,
        group_norm_num_groups=None,
        freeze_bn=False,
        freeze_bn_affine=False,
        projection=False,
        save_activations=False,
        scaler_rate = 1,
        pruning = False,
        need_scaler = False,
        global_rate = 1,
    ):
        super(ResNet_imagenet, self).__init__()
        self.dataset = dataset
        self.freeze_bn = freeze_bn
        self.freeze_bn_affine = freeze_bn_affine
        self.need_scaler = need_scaler

        track_stat = not self.freeze_bn

        model_params = decide_model_params(pruning)
        block_fn = model_params[resnet_size]["block"]
        block_nums = model_params[resnet_size]["layers"]

        # decide the num of classes.
        self.num_classes = self._decide_num_classes()

        # define layers.
        self.inplanes = int(round(64 * scaler_rate))
        # self.conv1 = nn.Conv2d(
        #     in_channels=3,
        #     out_channels=64,
        #     kernel_size=7,
        #     stride=2,
        #     padding=3,
        #     bias=False,
        # )

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        if self.need_scaler:
            self.scaler = Scaler(scaler_rate / global_rate)
        self.bn1 = norm2d(group_norm_num_groups, planes=self.inplanes, track_running_stats = track_stat)
        self.relu = nn.ReLU(inplace=True)

        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_block(
            block_fn=block_fn,
            planes=64,
            stride=1,
            block_num=block_nums[0],
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_stat,
            scaler_rate = scaler_rate,
            global_rate=global_rate
        )
        self.layer2 = self.make_block(
            block_fn=block_fn,
            planes=128,
            block_num=block_nums[1],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_stat,
            scaler_rate=scaler_rate,
            global_rate=global_rate
        )
        self.layer3 = self.make_block(
            block_fn=block_fn,
            planes=256,
            block_num=block_nums[2],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_stat,
            scaler_rate=scaler_rate,
            global_rate=global_rate
        )
        self.layer4 = self.make_block(
            block_fn=block_fn,
            planes=512,
            block_num=block_nums[3],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_stat,
            scaler_rate=scaler_rate,
            global_rate=global_rate
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = projection

        if self.projection:
            self.projection_layer = nn.Sequential(
                nn.Linear(int(round(512 * block_fn.expansion * scaler_rate)), int(round(512 * block_fn.expansion * scaler_rate))),
                nn.ReLU(),
                nn.Linear(int(round(512 * block_fn.expansion* scaler_rate)), 256)
            )
            self.classifier = nn.Linear(
                in_features=256,
                out_features=self.num_classes,
            )
        else:
            self.classifier = nn.Linear(
                in_features=int(round(512 * block_fn.expansion * scaler_rate)), out_features=self.num_classes
            )
        self.save_activations = save_activations
        # weight initialization based on layer type.
        self._weight_initialization()
        self.train()

    def forward(self, x,start_layer_idx = 0):
        if start_layer_idx >= 0:
            x = self.conv1(x)
            if self.need_scaler:
                x = self.scaler(x)
            x = self.bn1(x)
            x = self.relu(x)

            #x = self.maxpool(x)
            x = self.layer1(x)
            activation1 = x
            x = self.layer2(x)
            activation2 = x
            x = self.layer3(x)
            activation3 = x
            x = self.layer4(x)
            activation4 = x
            x = self.avgpool(x)
            feature = x.view(x.size(0), -1)
            if self.projection:
                feature = self.projection_layer(feature)

            if self.save_activations:
                self.activations = [activation1, activation2, activation3,activation4]
        else:
            feature = x
        x = self.classifier(feature)
        return x

    def make_block(self,block_fn,planes,block_num,stride = 1,
                    group_norm_num_groups=None,
                    track_running_stats=True,
                    scaler_rate = 1,
                    global_rate = 1):
        downsample = None
        if stride != 1 or self.inplanes != int(round(planes * block_fn.expansion * scaler_rate)):
            if self.need_scaler:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        int(round(planes * block_fn.expansion * scaler_rate)) ,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    Scaler(scaler_rate / global_rate),
                    norm2d(group_norm_num_groups, planes=int(round(planes * block_fn.expansion * scaler_rate)),
                           track_running_stats=track_running_stats),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        int(round(planes * block_fn.expansion * scaler_rate)) ,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    norm2d(group_norm_num_groups, planes=int(round(planes * block_fn.expansion* scaler_rate)) ,
                           track_running_stats=track_running_stats),
                )

        layers = []
        layers.append(
            block_fn(
                in_planes=self.inplanes,
                out_planes= planes * scaler_rate,
                stride=stride,
                downsample=downsample,
                group_norm_num_groups=group_norm_num_groups,
                track_running_stats=track_running_stats,
                rate = scaler_rate,
                need_scaler = self.need_scaler,
                global_rate = global_rate
            )
        )
        self.inplanes = int(round(planes * block_fn.expansion* scaler_rate))

        for _ in range(1, block_num):
            layers.append(
                block_fn(
                    in_planes=self.inplanes,
                    out_planes= planes * scaler_rate,
                    group_norm_num_groups=group_norm_num_groups,
                    track_running_stats=track_running_stats,
                    rate = scaler_rate,
                    need_scaler=self.need_scaler,
                    global_rate=global_rate
                )
            )
        return nn.Sequential(*layers)

class ResNet_cifar(ResNetBase):
    def __init__(
        self,
        dataset,
        resnet_size,
        scaling=1,
        save_activations=False,
        group_norm_num_groups=None,
        freeze_bn=False,
        freeze_bn_affine=False,
    ):
        super(ResNet_cifar, self).__init__()
        self.dataset = dataset
        self.freeze_bn = freeze_bn
        self.freeze_bn_affine = freeze_bn_affine

        # define model.
        if resnet_size % 6 != 2:
            raise ValueError("resnet_size must be 6n + 2:", resnet_size)
        block_nums = (resnet_size - 2) // 6
        block_fn = Bottleneck if resnet_size >= 44 else BasicBlock

        # decide the num of classes.
        self.num_classes = self._decide_num_classes()

        # define layers.
        assert int(16 * scaling) > 0
        self.inplanes = int(16 * scaling)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=(16 * scaling),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=int(16 * scaling))
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=int(16 * scaling),
            block_num=block_nums,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=int(32 * scaling),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=int(64 * scaling),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.classifier = nn.Linear(
            in_features=int(64 * scaling * block_fn.expansion),
            out_features=self.num_classes,
        )

        # weight initialization based on layer type.
        self._weight_initialization()

        # a placeholder for activations in the intermediate layers.
        self.save_activations = save_activations
        self.activations = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        activation1 = x
        x = self.layer2(x)
        activation2 = x
        x = self.layer3(x)
        activation3 = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if self.save_activations:
            self.activations = [activation1, activation2, activation3]
        return x

class CifarResNet(ResNetBase):
    def __init__(
        self,
        dataset,
        resnet_size,
        scaling=1,
        save_activations=False,
        group_norm_num_groups=None,
        freeze_bn=False,
        freeze_bn_affine=False,
        projection = False
    ):
        super(CifarResNet, self).__init__()

        self.dataset = dataset
        self.freeze_bn = freeze_bn
        self.freeze_bn_affine = freeze_bn_affine

        # define model.
        if resnet_size % 6 != 2:
            raise ValueError("resnet_size must be 6n + 2:", resnet_size)
        block_nums = (resnet_size - 2) // 6
        block_fn = Bottleneck if resnet_size >= 44 else BasicBlock

        # decide the num of classes.
        self.num_classes = self._decide_num_classes()


        # define layers.
        assert int(16 * scaling) > 0
        self.inplanes = int(16 * scaling)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=(16 * scaling),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=int(16 * scaling))
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=int(16 * scaling),
            block_num=block_nums,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=int(32 * scaling),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=int(64 * scaling),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        feature_dim = int(64 * scaling * block_fn.expansion)
        self.projection = projection
        if self.projection:

            self.projection_layer = nn.Sequential(
                nn.Linear(feature_dim,feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim,256)
            )
            self.classifier = nn.Linear(
                in_features=256,
                out_features=self.num_classes,
            )
        else:
            self.classifier = nn.Linear(
                in_features=feature_dim,
                out_features=self.num_classes,
            )
        # weight initialization based on layer type.
        self._weight_initialization()

        # a placeholder for activations in the intermediate layers.
        self.save_activations = save_activations
        self.activations = None

    def forward(self, x, start_layer_idx = 0):
        if start_layer_idx >= 0:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            feature = x
            if self.projection:
                feature = self.projection_layer(feature)

        else:
            feature = x
        x = self.classifier(feature)

        return x

from pcode.models.lowrank_resnet import FullRankBottleneck
import re

def resnet(conf, arch=None):
    factor = 1

    if len(arch.split('_')) > 1:
        arch, factor = arch.split('_')
        factor = eval(factor)

    resnet_size = int(re.findall(r"\D.([1-9].)", (arch if arch is not None else conf.arch))[0])
    dataset = conf.data
    save_activations = True if conf.AT_beta > 0 else False

    if factor <= 1:
        model = ResNet_imagenet(
            dataset=dataset,
            resnet_size=resnet_size,
            group_norm_num_groups=conf.group_norm_num_groups,
            freeze_bn=conf.freeze_bn,
            freeze_bn_affine=conf.freeze_bn_affine,
            projection=conf.projection,
            save_activations=save_activations,
            scaler_rate=factor,
            pruning=conf.pruning,
            need_scaler=conf.need_scaler,
            global_rate=conf.global_rate
        )
    elif conf.low_rank or factor > 1:
        model_params = {
            18: {"block": [LowRankBasicBlockConv1x1,FullRankBasicBlock], "layers": [2, 2, 2, 2]},
            34: {"block": [LowRankBasicBlockConv1x1,FullRankBasicBlock], "layers": [3, 4, 6, 3]},
            50: {"block": [LowRankBottleneckConv1x1,FullRankBottleneck], "layers": [3, 4, 6, 3]},
            101: {"block": [LowRankBottleneckConv1x1,FullRankBottleneck], "layers": [3, 4, 23, 3]},
            152: {"block": [LowRankBottleneckConv1x1,FullRankBottleneck], "layers": [3, 8, 36, 3]},
        }
        block = model_params[resnet_size]["block"]
        layers = model_params[resnet_size]["layers"]
        model = HybridResNet(block[0],block[1],rank_factor=factor,layers=layers,
                             num_classes=decide_num_classes(dataset),track_running_stats = not conf.freeze_bn)
    else:
        raise NotImplementedError

    return model


if __name__ == "__main__":
    import torch

    print("cifar10")
    net = ResNet_cifar(
        dataset="cifar10",
        resnet_size=20,
        group_norm_num_groups=2,
        freeze_bn=True,
        freeze_bn_affine=True,
    )
    print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.shape)

    # print("imagenet")
    # net = ResNet_imagenet(
    #     dataset="imagenet", resnet_size=50, group_norm_num_groups=None
    # )
    # print(net)
    # x = torch.randn(1, 3, 224, 224)
    # y = net(x)
    # print(y.shape)
