import copy
import logging
import numpy as np
import operator
from collections import OrderedDict
from typing import Union, List, Type
from .resnet import resnet
import torch
import torch.nn as nn
import torch.nn.functional as func

from .bn_ops import get_bn_layer, DualNormLayer
from .slimmable_ops import SlimmableConv2d, SlimmableLinear, SwitchableLayer1D, \
    SlimmableBatchNorm2d, SlimmableBatchNorm1d, SlimmableOpMixin

class BaseModule(nn.Module):
    def set_bn_mode(self, is_noised: Union[bool, torch.Tensor]):
        """Set BN mode to be noised or clean. This is only effective for StackedNormLayer
        or DualNormLayer."""
        def set_bn_eval_(m):
            if isinstance(m, (DualNormLayer,)):
                if isinstance(is_noised, (float, int)):
                    m.clean_input = 1. - is_noised
                elif isinstance(is_noised, torch.Tensor):
                    m.clean_input = ~is_noised
                else:
                    m.clean_input = not is_noised
        self.apply(set_bn_eval_)

    # forward
    def forward(self, x):
        z = self.encode(x)
        logits = self.decode_clf(z)
        return logits

    def encode(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        z = torch.flatten(x, 1)
        return z

    def decode_clf(self, z):
        logits = self.classifier(z)
        return logits

    def mix_dual_forward(self, x, lmbd, deep_mix=False):
        if deep_mix:
            self.set_bn_mode(lmbd)
            logit = self.forward(x)
        else:
            # FIXME this will result in unexpected result for non-dual models?
            logit = 0
            if lmbd < 1:
                self.set_bn_mode(False)
                logit = logit + (1 - lmbd) * self.forward(x)

            if lmbd > 0:
                self.set_bn_mode(True)
                logit = logit + lmbd * self.forward(x)
        return logit


def get_slim_ratios_from_str(mode: str, sort=True):
    if mode.startswith('ln'):  # lognorm
        base_width = 0.125
        ws = np.arange(0., 1. + base_width, base_width)
        slimmable_ratios = ws[1:]
        return slimmable_ratios

    ps = mode.split('-')
    slimmable_ratios = []
    for p in ps:
        if 'd' in p:
            p, q = p.split('d')  # p: 1/p-net; q: weight of the net in samples
            p, q = int(p), int(q)
            p = p * 1. / q
        else:
            p = int(p)
        slimmable_ratios.append(1. / p)
    if sort:
        slimmable_ratios = sorted(slimmable_ratios)
    return slimmable_ratios


def parse_lognorm_slim_schedule(train_slim_ratios, mode, client_num):
    ws = sorted(train_slim_ratios)
    min_w = min(train_slim_ratios)
    from scipy.stats import lognorm
    s, scale = [float(v) for v in mode[len('ln'):].split('_')]
    rv = lognorm(s=s, scale=scale)
    # print(ws)
    cdfs = [rv.cdf(w) for w in ws] + [1.]
    # print(cdfs)
    qs = [c - rv.cdf(min_w) for c in cdfs]
    r = (qs[-1] - qs[0])
    qs = [int(client_num * (q - qs[0]) / r) for q in qs]
    # print(qs)
    slim_ratios = np.zeros(client_num)
    for i in range(len(qs) - 1):
        slim_ratios[qs[i]:qs[i + 1]] = ws[i]
    return slim_ratios


class SlimmableMixin(object):
    # define the candidate model widths for customization
    #slimmable_ratios = [0.125, 0.25, 0.5, 1.0]
    slimmable_ratios = [0.35, 0.7, 1.05, 3.15]

    def _set_slimmabe_ratios(self, mode: Union[str, List], sort=True):
        """Define the slim_ratio for groups. For example, 8-4-2-1 [default]
        means x1/8 net for the 1st group, and x1/4 for the 2nd."""
        if isinstance(mode, str):
            self.slimmable_ratios = get_slim_ratios_from_str(mode, sort)
            print(f"Set model slim ratios: {self.slimmable_ratios} by mode: {mode}")
        else:
            if mode is not None:
                self.slimmable_ratios = mode
            if sort:
                self.slimmable_ratios = sorted(self.slimmable_ratios)
            print(f"Set model slim ratios: {self.slimmable_ratios}")
        return self.slimmable_ratios

    def switch_slim_mode(self, ratio, slim_bias_idx=0, out_slim_bias_idx=None,
                         mix_forward_num=1):
        # print(f"### switch ratio {ratio}, slim_bias_idx {slim_bias_idx}, "
        #       f"out_slim_bias_idx {out_slim_bias_idx}, mix_forward_num {mix_forward_num}")
        if ratio is None:
            return
        for m in list(self.modules()) + [self]:
            if isinstance(m, SwitchableLayer1D):
                assert ratio in self.slimmable_ratios, \
                    f"Since SwitchableLayer1D is used, only ratios from {self.slimmable_ratios} are allowed. " \
                    f"But get ratio={ratio}"
            if hasattr(m, 'slim_ratio'):
                m.slim_ratio = ratio
                m.slim_bias_idx = slim_bias_idx
                m.out_slim_bias_idx = out_slim_bias_idx
                m.mix_forward_num = mix_forward_num


class Ensemble(nn.Module):
    def __init__(self, full_net: SlimmableMixin, weights=None):
        super().__init__()
        assert isinstance(full_net, SlimmableMixin)
        self.full_net = full_net
        if isinstance(weights, (list, tuple)):
            weights = torch.tensor(weights).cuda()
            weights = weights / torch.sum(weights)
            print(f" Subnets weights: {weights}")
        self.weights = weights

    @property
    def input_shape(self):
        return self.full_net.input_shape

    def set_total_slim_ratio(self, r):
        assert r <= self._max_total_slim_ratio, f"try to set total_slim_ratio as {r}," \
                                                f" but the max value should be {self._max_total_slim_ratio}"
        self._total_slim_ratio = r

    def _reduce_subnet_logits(self, all_logits, shift_reduction):
        if shift_reduction == 'mean':
            all_logits = torch.stack(all_logits, dim=-1)
            if self.weights is not None:
                all_logits = all_logits * torch.reshape(self.weights, (1, 1, -1))
                logits = torch.sum(all_logits, dim=-1)
            else:
                logits = torch.mean(all_logits, dim=-1)
        elif shift_reduction == 'cat':
            logits = torch.cat(all_logits, dim=1)
        elif shift_reduction == 'stack':
            logits = torch.stack(all_logits, dim=-1)
        else:
            raise ValueError(f"Invalid shift_reduction: {shift_reduction}")
        return logits


class EnsembleGroupSubnet(Ensemble):
    """Ensemble subnet in a given big net."""
    def __init__(self, full_net: SlimmableMixin, subnet_ratios=[0.125,0.25], shift_idxs=[0, 0], weights=None):
        super(EnsembleGroupSubnet, self).__init__(full_net, weights=weights)
        self.subnet_ratios = subnet_ratios
        if np.isscalar(shift_idxs):
            shift_idxs = [shift_idxs] * len(subnet_ratios)
        self.shift_idxs = shift_idxs
        assert len(shift_idxs) == len(subnet_ratios), f"Length not match. len(shift_idxs)={len(shift_idxs)}" \
                                                      f" while len(subnet_ratios)= {len(subnet_ratios)}"
        self._total_slim_ratio = sum(subnet_ratios)
        self._max_total_slim_ratio = sum(subnet_ratios)

    def forward(self, x, subnet_reduction='mean', **kwargs):
        # logits = 0.
        ensemble_num = 0
        _subnet_ratios, _shift_idxs = self.get_current_subnet_bounds()
        all_logits = []
        all_feas = []
        for subnet_ratio, shift_idx in zip(_subnet_ratios, _shift_idxs):
            self.full_net.switch_slim_mode(subnet_ratio, slim_bias_idx=shift_idx)
            if 'return_pre_clf_fea' in kwargs:
                logits, feas = self.full_net(x, **kwargs)
                all_feas.append(feas)
            else:
                logits = self.full_net(x, **kwargs)
            all_logits.append(logits)
            ensemble_num += 1
        logits = self._reduce_subnet_logits(all_logits, subnet_reduction)
        if 'return_pre_clf_fea' in kwargs:
            feas = self._reduce_subnet_logits(all_feas, subnet_reduction)
            return logits, feas
        else:
            return logits

    def get_current_subnet_bounds(self):
        if self._total_slim_ratio < self._max_total_slim_ratio:
            _subnet_ratios = []
            _shift_idxs = []
            s = 0
            for ir, r in enumerate(self.subnet_ratios):
                s += r
                if s <= self._total_slim_ratio:
                    _subnet_ratios.append(r)
                    _shift_idxs.append(self.shift_idxs[ir])
        else:
            _subnet_ratios = self.subnet_ratios
            _shift_idxs = self.shift_idxs
        return _subnet_ratios, _shift_idxs

    def state_size(self):
        """Return model size based on state_dict."""
        _subnet_ratios, _shift_idxs = self.get_current_subnet_bounds()
        size = 0
        for subnet_ratio, shift_idx in zip(_subnet_ratios, _shift_idxs):
            self.full_net.switch_slim_mode(subnet_ratio, slim_bias_idx=shift_idx)
            size += count_params_by_state(self.full_net)
        return size


class EnsembleSubnet(Ensemble):
    """Ensemble subnet in a given big net."""
    def __init__(self, full_net: SlimmableMixin, subnet_ratio=0.125, ensemble_num=-1, shift_idx=0,
                 weights=None):
        super(EnsembleSubnet, self).__init__(full_net, weights=weights)
        self.subnet_ratio = subnet_ratio
        self.shift_idx = shift_idx
        if ensemble_num < 0:
            ensemble_num = int(1/self.subnet_ratio)
        self.ensemble_num = ensemble_num
        self._total_slim_ratio = subnet_ratio * self.ensemble_num
        self._max_total_slim_ratio = subnet_ratio * self.ensemble_num

    def forward(self, x, subnet_reduction='mean', **kwargs):
        if self._total_slim_ratio < self._max_total_slim_ratio:
            ensemble_num = int(self._total_slim_ratio / self.subnet_ratio)
        else:
            ensemble_num = self.ensemble_num
        all_logits = []
        all_feas = []
        for shift_idx in range(self.shift_idx, self.shift_idx+ensemble_num):
            self.full_net.switch_slim_mode(self.subnet_ratio, slim_bias_idx=shift_idx)
            if 'return_pre_clf_fea' in kwargs:
                logits, feas = self.full_net(x, **kwargs)
                all_feas.append(feas)
            else:
                logits = self.full_net(x, **kwargs)
            all_logits.append(logits)
        logits = self._reduce_subnet_logits(all_logits, subnet_reduction)
        if 'return_pre_clf_fea' in kwargs:
            feas = self._reduce_subnet_logits(all_feas, subnet_reduction)
            return logits, feas
        else:
            return logits

    def state_size(self):
        """Return model size based on state_dict."""
        if self._total_slim_ratio < self._max_total_slim_ratio:
            ensemble_num = int(self._total_slim_ratio / self.subnet_ratio)
        else:
            ensemble_num = self.ensemble_num
        size = 0
        for shift_idx in range(self.shift_idx, self.shift_idx+ensemble_num):
            self.full_net.switch_slim_mode(self.subnet_ratio, slim_bias_idx=shift_idx)
            size += count_params_by_state(self.full_net)
        return size

    def forward_with_layer_features(self, x):
        logits = 0.
        all_features = []
        if self._total_slim_ratio < self._max_total_slim_ratio:
            ensemble_num = int(self._total_slim_ratio / self.subnet_ratio)
        else:
            ensemble_num = self.ensemble_num
        for shift_idx in range(self.shift_idx, self.shift_idx+ensemble_num):
            self.full_net.switch_slim_mode(self.subnet_ratio, slim_bias_idx=shift_idx)
            l, features = self.full_net.forward_with_layer_features(x)
            logits = logits + l
            if len(all_features) == 0:
                all_features = features
            else:
                for i, f1, f2 in zip(range(len(all_features)), all_features, features):
                    # if i > 0: # ad-hoc non-slimmable first layer. Thus we skip the first layer here.
                    all_features[i] = torch.cat((f1, f2), dim=1)
        return logits / ensemble_num, all_features


class EnsembleNet(BaseModule, SlimmableMixin):
    def __init__(self, conf, base_net: Type, arch, bn_type='bn', width_scale=1, atom_slim_ratio=0.125, slimmable_ratios=None):
        super(EnsembleNet, self).__init__()

        self._set_slimmabe_ratios(slimmable_ratios)
        self.atom_slim_ratio = min(min(slimmable_ratios), conf.atom_slim_ratio)

        num_ens = round(max(self.slimmable_ratios) / self.atom_slim_ratio)
        self.base_net = base_net
        self.bn_type = bn_type

        #conf.pruning = True
        #arch = arch.split('_')[0]
        if min(slimmable_ratios) == self.atom_slim_ratio:
            min_arch = conf.arch_info["worker"][-1]
        else:
            min_arch = conf.arch_info["worker"][-1].split("_")[0] + "_" + str(self.atom_slim_ratio)
        #print(min_arch, num_ens)
        for i in range(num_ens):
            self.add_module(str(i), self.base_net(conf, arch = min_arch))
            # self.add_module(str(i), self.base_net(
            #     num_classes=num_classes, track_running_stats=track_running_stats,
            #     bn_type=bn_type, share_affine=share_affine, #slimmable_layers=slimmable_layers,
            #     width_scale=width_scale*atom_slim_ratio, **kwargs))
        #conf.pruning = False

        self.slim_bias_idx = 0
        self.slim_ratio = max(self.slimmable_ratios)
        if width_scale != 1.:
            raise NotImplementedError()
            # the scale is to keep the same training widths.
            self.slimmable_ratios = [r/width_scale for r in self.slimmable_ratios]

        self._max_total_slim_ratio = 1.
        self.out_slim_bias_idx = self.slim_ratio
        self.mix_forward_num = 0
        # self._total_slim_ratio = self.slim_ratio
        self.base_idxs = list(range(num_ens))

    @property
    def input_shape(self):
        return self._modules['0'].input_shape

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx: int):
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx: int, module) -> None:
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def __len__(self) -> int:
        return len(self._modules)

    def forward(self, x):
        base_idxs = self.current_slice()
        logits = [self[i](x) for i in base_idxs]
        if len(base_idxs) > 1:
            logits = torch.mean(torch.stack(logits, dim=-1), dim=-1)
        else:
            logits = logits[0]

        return logits

    def current_slice(self):
        start = self.slim_bias_idx
        # end = start + int(len(self) * self.slim_ratio)
        end = start + round(self.slim_ratio / self.atom_slim_ratio)
        #print(start,end)
        assert end <= len(self), "Invalid slim_ratio. Too many subnets required. " \
                                 f"Have {len(self)} but require " \
                                 f"{end}-{start}={end-start}"
        return self.base_idxs[start:end]

    # make this more like an Ensemble class.
    @property
    def full_net(self):
        return self

    def set_total_slim_ratio(self, r):
        assert r <= self._max_total_slim_ratio, f"try to set total_slim_ratio as {r}," \
                                                f" but the max value should be {self._max_total_slim_ratio}"
        # self._total_slim_ratio = r
        self.slim_ratio = r

    def state_dict(self, full_size=False, destination=None, prefix='', keep_vars=False):
        """full_size: get full width state_dict. By default, return current width state."""
        if full_size:
            return super(EnsembleNet, self).state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars)
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = local_metadata = dict(version=self._version)
        self._save_to_state_dict(destination, prefix, keep_vars)
        base_idxs = self.current_slice()
        for idx in base_idxs:  # self._modules.items():
            name = self._get_abs_string_index(idx)
            module = self._modules[name]
            if module is not None:
                module.state_dict(destination, prefix + name + '.', keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination


class SlimmableDigitModel(BaseModule, SlimmableMixin):
    """
    Model for benchmark experiment on Digits.
    """
    input_shape = [None, 3, 28, 28]

    def __init__(self, num_classes=10, bn_type='bn', track_running_stats=True, slimmabe_ratios=None):
        super(SlimmableDigitModel, self).__init__()
        self._set_slimmabe_ratios(slimmabe_ratios)
        bn_class = get_bn_layer(bn_type)
        self.bn_type = bn_type

        self.conv1 = SlimmableConv2d(3, 64, 5, 1, 2, non_slimmable_in=True)
        if track_running_stats:
            self.bn1 = SwitchableLayer1D(bn_class['2d'], 64, slim_ratios=self.slimmable_ratios,
                                         track_running_stats=track_running_stats)
        else:
            assert bn_type == 'bn', "for now, we can only use BN."
            self.bn1 = SlimmableBatchNorm2d(64,  track_running_stats=track_running_stats)
        self.conv2 = SlimmableConv2d(64, 64, 5, 1, 2)
        if track_running_stats:
            self.bn2 = SwitchableLayer1D(bn_class['2d'], 64, slim_ratios=self.slimmable_ratios,
                                         track_running_stats=track_running_stats)
        else:
            assert bn_type == 'bn', "for now, we can only use BN."
            self.bn2 = SlimmableBatchNorm2d(64,  track_running_stats=track_running_stats)
        self.conv3 = SlimmableConv2d(64, 128, 5, 1, 2)
        if track_running_stats:
            self.bn3 = SwitchableLayer1D(bn_class['2d'], 128, slim_ratios=self.slimmable_ratios,
                                         track_running_stats=track_running_stats)
        else:
            assert bn_type == 'bn', "for now, we can only use BN."
            self.bn3 = SlimmableBatchNorm2d(128, track_running_stats=track_running_stats)

        self.fc1 = SlimmableLinear(128*7*7, 2048)
        if track_running_stats:
            self.bn4 = SwitchableLayer1D(bn_class['1d'], 2048, slim_ratios=self.slimmable_ratios,
                                         track_running_stats=track_running_stats)
        else:
            assert bn_type == 'bn', "for now, we can only use BN."
            self.bn4 = SlimmableBatchNorm1d(2048,  track_running_stats=track_running_stats)
        self.fc2 = SlimmableLinear(2048, 512)
        if track_running_stats:
            self.bn5 = SwitchableLayer1D(bn_class['1d'], 512, slim_ratios=self.slimmable_ratios,
                                         track_running_stats=track_running_stats)
        else:
            assert bn_type == 'bn', "for now, we can only use BN."
            self.bn5 = SlimmableBatchNorm1d(512,  track_running_stats=track_running_stats)
        self.fc3 = SlimmableLinear(512, num_classes, non_slimmable_out=True)

    def forward(self, x):
        z = self.encode(x)
        return self.decode_clf(z)

    def encode(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)
        return x

    def decode_clf(self, x):
        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = func.relu(x)

        logits = self.fc3(x)
        return logits


class SlimmableAlexNet(BaseModule, SlimmableMixin):
    """
    used for DomainNet and Office-Caltech10
    """
    input_shape = [None, 3, 256, 256]

    def __init__(self, num_classes=10, track_running_stats=True, bn_type='bn', share_affine=True,
                 width_scale=1., slimmabe_ratios=None):
        super(SlimmableAlexNet, self).__init__()
        self._set_slimmabe_ratios(slimmabe_ratios)
        self.bn_type = bn_type
        bn_class = get_bn_layer(bn_type)
        # share_affine
        bn_kwargs = dict(
            track_running_stats=track_running_stats,
        )
        if bn_type.startswith('d'):  # dual BN
            bn_kwargs['share_affine'] = share_affine
        if track_running_stats:
            norm_layer2d = lambda ch: SwitchableLayer1D(bn_class['2d'], ch, slim_ratios=self.slimmable_ratios, **bn_kwargs)
            norm_layer1d = lambda ch: SwitchableLayer1D(bn_class['1d'], ch, slim_ratios=self.slimmable_ratios, **bn_kwargs)
        else:
            assert bn_type == 'bn'
            norm_layer2d = lambda ch: SlimmableBatchNorm2d(ch, affine=True, **bn_kwargs)
            norm_layer1d = lambda ch: SlimmableBatchNorm1d(ch, affine=True, **bn_kwargs)
        feature_layers = []
        feature_layers += [
            ('conv1', SlimmableConv2d(3, int(64*width_scale), kernel_size=11, stride=4, padding=2,
                                      non_slimmable_in=True)),
            ('bn1', norm_layer2d(int(64*width_scale))),
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

            ('conv2', SlimmableConv2d(int(64*width_scale), int(192*width_scale), kernel_size=5, padding=2)),
            ('bn2', norm_layer2d(int(192*width_scale))),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

            ('conv3', SlimmableConv2d(int(192*width_scale), int(384*width_scale), kernel_size=3, padding=1)),
            ('bn3', norm_layer2d(int(384*width_scale))),
            ('relu3', nn.ReLU(inplace=True)),

            ('conv4', SlimmableConv2d(int(384*width_scale), int(256*width_scale), kernel_size=3, padding=1)),
            ('bn4', norm_layer2d(int(256*width_scale))),
            ('relu4', nn.ReLU(inplace=True)),

            ('conv5', SlimmableConv2d(int(256*width_scale), int(256*width_scale), kernel_size=3, padding=1)),
            ('bn5', norm_layer2d(int(256*width_scale))),
            ('relu5', nn.ReLU(inplace=True)),
            ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
        ]
        self.features = nn.Sequential(
            OrderedDict(feature_layers)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        clf_layers = [
            ('fc1', SlimmableLinear(int(256 * 6 * 6*width_scale), int(4096*width_scale))),
            ('bn6', norm_layer1d(int(4096*width_scale))),
            ('relu6', nn.ReLU(inplace=True)),

            ('fc2', SlimmableLinear(int(4096*width_scale), int(4096*width_scale))),
            ('bn7', norm_layer1d(int(4096*width_scale))),
            ('relu7', nn.ReLU(inplace=True)),

            ('fc3', SlimmableLinear(int(4096*width_scale), num_classes, non_slimmable_out=True)),
        ]

        self.classifier = nn.Sequential(
            OrderedDict(clf_layers)
        )

    def get_module_by_layer(self):
        # layers
        blocks = [
            [self.features._modules[name] for name in ['conv1', 'bn1', 'relu1', 'maxpool1']],
            [self.features._modules[name] for name in ['conv2', 'bn2', 'relu2', 'maxpool2']],
            [self.features._modules[name] for name in ['conv3', 'bn3', 'relu3']],
            [self.features._modules[name] for name in ['conv4', 'bn4', 'relu4']],
            [self.features._modules[name] for name in ['conv5', 'bn5', 'relu5', 'maxpool5']],
            [self.avgpool, nn.Flatten()],
            [self.classifier._modules[name] for name in ['fc1', 'bn6', 'relu6']],
            [self.classifier._modules[name] for name in ['fc2', 'bn7', 'relu7']],
            [self.classifier._modules[name] for name in ['fc3']],
        ]
        return blocks

    def print_footprint(self):
        input_shape = self.input_shape
        input_shape[0] = 2
        x = torch.rand(input_shape)
        batch = x.shape[0]
        layers = self.get_module_by_layer()
        print(f"input: {x.shape[1:]} => {np.prod(x.shape[1:])}")
        for i_layer, layer in enumerate(layers):
            for m in layer:
                x = m(x)
            print(f"layer {i_layer}: {np.prod(x.shape[1:]):5d} <= {x.shape[1:]}")


from thop import profile
def count_params_by_state(model):
    """Count #param based on state dict of the given model."""
    if hasattr(model, 'state_size'):  # EnsembleSubnet, EnsembleGroupSubnet
        s = model.state_size()
    else:
        s = 0
        for k, p in model.state_dict().items():
            s = s + p.numel()
    return s

def profile_model(model, verbose=False, batch_size=2, device='cpu', input_shape=None):
    if input_shape is None:
        input_shape = model.input_shape
    input_shape = (batch_size, *input_shape[1:])
    dummy_input = torch.rand(input_shape).to(device)
    # customized ops:
    #       https://github.com/Lyken17/pytorch-OpCounter/blob/master/thop/vision/basic_hooks.py
    state_params = count_params_by_state(model)
    flops, params = profile(model, inputs=(dummy_input,), custom_ops=thop_hooks,
                            verbose=verbose)
    flops = flops / batch_size
    return flops, state_params

def profile_slimmable_models(model, slim_ratios, verbose=1):
    max_flops = None
    max_params = None
    for slim_ratio in sorted(slim_ratios, reverse=True):
        if hasattr(model, 'switch_slim_mode'):
            model.switch_slim_mode(slim_ratio)
        else:  # if isinstance(model, Ensemble):
            model.set_total_slim_ratio(slim_ratio)
        flops, state_params = profile_model(model, verbose > 1)
        if verbose > 0:
            print(f'slim_ratio: {slim_ratio:.3f} GFLOPS: {flops / 1e9:.4f}, '
                  f'model state size: {state_params / 1e6:.2f}MB')

        if max_flops is None:
            max_flops = flops
            max_params = state_params
        elif verbose > 0:
            print(f"    flop ratio: {flops/max_flops:.3f}, size ratio: {state_params/max_params:.3f},"
                  f" sqrt size ratio: {np.sqrt(state_params/max_params):.3f}")


def ensresnet(conf, arch=None):
    if conf.pruning:
        slimmable_ratios = [eval(arch.split('_')[-1]) for arch in conf.arch_info["worker"]]
    else:
        slimmable_ratios = [1.0 / float(eval(arch.split('_')[-1])) for arch in conf.arch_info["worker"]]

    model = EnsembleNet(base_net=resnet, arch = arch, conf = conf,
                        slimmable_ratios = slimmable_ratios)
    return model

def main():


    print(f"profile model GFLOPs (forward complexity) and size (#param)")

    model = SlimmableAlexNet(track_running_stats=False, bn_type='bn', share_affine=False)
    model.eval()  # this will affect bn etc

    print(f"model {model.__class__.__name__} on {'training' if model.training else 'eval'} mode")
    input_shape = model.input_shape
    # batch_size = 2
    # input_shape[0] = batch_size
    profile_slimmable_models(model, model.slimmable_ratios)
    print(f"\n==footprint==")
    model.switch_slim_mode(1.)
    model.print_footprint()
    print(f"\n==footprint==")
    model.switch_slim_mode(0.125)
    model.print_footprint()

    print(f'\n--------------')
    full_net = model
    model = EnsembleGroupSubnet(full_net, [0.125, 0.125, 0.25, 0.5], [0, 1, 1, 1])
    model.eval()
    print(f"model {model.__class__.__name__} on {'training' if model.training else 'eval'} mode")
    profile_slimmable_models(model, model.full_net.slimmable_ratios)

    print(f'\n--------------')
    model = EnsembleSubnet(full_net, 0.125)
    model.eval()
    print(f"model {model.__class__.__name__} on {'training' if model.training else 'eval'} mode")
    profile_slimmable_models(model, model.full_net.slimmable_ratios)


if __name__ == '__main__':
    main()
