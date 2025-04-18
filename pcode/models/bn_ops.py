import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.modules.batchnorm import _NormBase

class DualNormLayer(nn.Module):
    """Dual Normalization Layer."""
    _version = 1
    # __constants__ = ['track_running_stats', 'momentum', 'eps',
    #                  'num_features', 'affine']

    def __init__(self, num_features, track_running_stats=True, affine=True, bn_class=None,
                 share_affine=True, **kwargs):
        super(DualNormLayer, self).__init__()
        self.affine = affine
        if bn_class is None:
            bn_class = nn.BatchNorm2d
        self.bn_class = bn_class
        self.share_affine = share_affine
        self.clean_bn = bn_class(num_features, track_running_stats=track_running_stats, affine=self.affine and not self.share_affine, **kwargs)
        self.noise_bn = bn_class(num_features, track_running_stats=track_running_stats, affine=self.affine and not self.share_affine, **kwargs)
        if self.affine and self.share_affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.clean_input = True  # only used in training?

    def reset_parameters(self) -> None:
        if self.affine and self.share_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if isinstance(self.clean_input, bool):
            if self.clean_input:
                out = self.clean_bn(inp)
            else:
                out = self.noise_bn(inp)
        elif isinstance(self.clean_input, torch.Tensor):
            # Separate input. This important at training to avoid mixture of BN stats.
            clean_mask = torch.nonzero(self.clean_input)
            noise_mask = torch.nonzero(~self.clean_input)
            out = torch.zeros_like(inp)

            if len(clean_mask) > 0:
                clean_mask = clean_mask.squeeze(1)
                # print(self.clean_input, clean_mask)
                out_clean = self.clean_bn(inp[clean_mask])
                out[clean_mask] = out_clean
            if len(noise_mask) > 0:
                noise_mask = noise_mask.squeeze(1)
                # print(self.clean_input, noise_mask)
                out_noise = self.noise_bn(inp[noise_mask])
                out[noise_mask] = out_noise
        elif isinstance(self.clean_input, (float, int)):
            assert not self.training, "You should not use both BN at training."
            assert not self.share_affine, "Should not share affine, because we have to use affine" \
                                          " before combination but didn't."
            out_c = self.clean_bn(inp)
            out_n = self.noise_bn(inp)
            out = self.clean_input * 1. * out_c + (1. - self.clean_input) * out_n
        else:
            raise TypeError(f"Invalid self.clean_input: {type(self.clean_input)}")
        if self.affine and self.share_affine:
            # out = F.linear(out, self.weight, self.bias)
            shape = [1] * out.dim()
            shape[1] = -1
            out = out * self.weight.view(*shape) + self.bias.view(*shape)
            assert out.shape == inp.shape
            # TODO how to do the affine?
            # out = F.batch_norm(out, None, None, self.weight, self.bias, self.training)
        return out

# BN modules
class _MockBatchNorm(_NormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_MockBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return func.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            torch.zeros_like(self.running_mean),
            torch.ones_like(self.running_var),
            self.weight, self.bias, False, exponential_average_factor, self.eps)

class MockBatchNorm1d(_MockBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

class MockBatchNorm2d(_MockBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class BatchNorm2dAgent(nn.BatchNorm2d):
    def __init__(self, *args, log_stat=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_stat = None  # statistic before BN
        self.post_stat = None  # statistic after BN
        self.log_stat = log_stat

    def forward(self, input):
        if not self.log_stat:
            self.pre_stat = None
        else:
            self.pre_stat = {
                'mean': torch.mean(input, dim=[0, 2, 3]).data.cpu().numpy(),
                'var': torch.var(input, dim=[0, 2, 3]).data.cpu().numpy(),
                'data': input.data.cpu().numpy(),
            }
        out = super().forward(input)
        if not self.log_stat:
            self.pre_stat = None
        else:
            self.post_stat = {
                'mean': torch.mean(out, dim=[0,2,3]).data.cpu().numpy(),
                'var': torch.var(out, dim=[0,2,3]).data.cpu().numpy(),
                'data': out.data.cpu().numpy(),
                # 'mean': ((torch.mean(out, dim=[0, 2, 3]) - self.bias)/self.weight).data.cpu().numpy(),
                # 'var': (torch.var(out, dim=[0, 2, 3])/(self.weight**2)).data.cpu().numpy(),
            }
        return out

class BatchNorm1dAgent(nn.BatchNorm1d):
    def __init__(self, *args, log_stat=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_stat = None  # statistic before BN
        self.post_stat = None  # statistic after BN
        self.log_stat = log_stat

    def forward(self, input):
        if not self.log_stat:
            self.pre_stat = None
        else:
            self.pre_stat = {
                'mean': torch.mean(input, dim=[0]).data.cpu().numpy().copy(),
                'var': torch.var(input, dim=[0]).data.cpu().numpy().copy(),
                'data': input.data.cpu().numpy().copy(),
            }
        out = super().forward(input)
        if not self.log_stat:
            self.post_stat = None
        else:
            self.post_stat = {
                'mean': torch.mean(out, dim=[0]).data.cpu().numpy().copy(),
                'var': torch.var(out, dim=[0]).data.cpu().numpy().copy(),
                # 'mean': ((torch.mean(out, dim=[0]) - self.bias)/self.weight).data.cpu().numpy(),
                # 'var': (torch.var(out, dim=[0])/(self.weight**2)).data.cpu().numpy(),
                'data': out.detach().cpu().numpy().copy(),
            }
        # print("post stat mean: ", self.post_stat['mean'])
        return out


def is_film_dual_norm(bn_type: str):
    return bn_type.startswith('fd')


def get_bn_layer(bn_type: str):
    if bn_type.startswith('d'):  # dual norm layer. Example: sbn, sbin, sin
        base_norm_class = get_bn_layer(bn_type[1:])
        bn_class = {
            '1d': lambda num_features, **kwargs: DualNormLayer(num_features, bn_class=base_norm_class['1d'], **kwargs),
            '2d': lambda num_features, **kwargs: DualNormLayer(num_features, bn_class=base_norm_class['2d'], **kwargs),
        }
    elif is_film_dual_norm(bn_type):  # dual norm layer. Example: sbn, sbin, sin
        base_norm_class = get_bn_layer(bn_type[1:])
        bn_class = {
            '1d': lambda num_features, **kwargs: FilmDualNormLayer(num_features, bn_class=base_norm_class['1d'], **kwargs),
            '2d': lambda num_features, **kwargs: FilmDualNormLayer(num_features, bn_class=base_norm_class['2d'], **kwargs),
        }
    elif bn_type == 'bn':
        bn_class = {'1d': nn.BatchNorm1d, '2d': nn.BatchNorm2d}
    elif bn_type == 'none':
        bn_class = {'1d': MockBatchNorm1d,
                    '2d': MockBatchNorm2d}
    else:
        raise ValueError(f"Invalid bn_type: {bn_type}")
    return bn_class
