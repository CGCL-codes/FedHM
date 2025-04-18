from collections import OrderedDict
from pcode.models.slimmable_nets import ensresnet
import torch
import numpy as np
from parameters import get_args
import pcode.utils.param_parser as param_parser

conf = get_args()
conf.pruning=True
conf.arch_info = (
        param_parser.dict_parser(conf.complex_arch)
        if conf.complex_arch is not None
        else {"master": conf.arch, "worker": conf.arch}
    )
conf.data="tiny-imagenet"
conf.num_classes=200
conf.arch_info["worker"] = ["ensresnet34_2.8","ensresnet34_0.4"]
conf.atom_slim_ratio = 0.4
master_model = ensresnet(conf)
idx_i = None
param_idx = OrderedDict()
scaler_rate = 0.5
def param_counter(model):
    num_params = 0
    for param_index, (param_name, param) in enumerate(model.named_parameters()):
        num_params += param.numel()
    return num_params / 1e6

client_model = master_model
#client_model = ResNet_imagenet("cifar10",18,freeze_bn=True,scaler_rate=float(1/8))
print("@@@ Num Params: Vanilla Model: {}, Hybrid Model: {}".format(param_counter(master_model),
                                                                         param_counter(client_model)))

# for k, v in master_model.state_dict().items():
#     parameter_type = k.split('.')[-1]
#
#
#     if 'weight' in parameter_type or 'bias' in parameter_type:
#         if parameter_type == 'weight':
#             if v.dim() > 1:
#                 input_size = v.size(1)
#                 output_size = v.size(0)
#                 if 'conv1' in k or 'conv2' in k:
#                     if idx_i is None:
#                         idx_i = torch.arange(input_size, device=v.device)
#                     input_idx_i_m = idx_i
#                     # scaler_rate = self.model_rate[archs[m]]
#                     local_output_size = int(np.ceil(output_size * scaler_rate))
#                     output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
#                     idx_i = output_idx_i_m
#                 elif 'downsample.0' in k:
#                     input_idx_i_m = param_idx[k.replace('downsample.0', 'conv1')][1]
#                     output_idx_i_m = idx_i
#                 elif 'classifier' in k:
#                     input_idx_i_m = idx_i
#                     output_idx_i_m = torch.arange(output_size, device=v.device)
#                 else:
#                     raise ValueError('Not valid k')
#                 param_idx[k] = (output_idx_i_m, input_idx_i_m)
#             else:
#                 input_idx_i_m = idx_i
#                 param_idx[k] = input_idx_i_m
#         else:
#             input_size = v.size(0)
#             if 'classifier' in k:
#                 input_idx_i_m = torch.arange(input_size, device=v.device)
#                 param_idx[k] = input_idx_i_m
#             else:
#                 input_idx_i_m = idx_i
#                 param_idx[k] = input_idx_i_m
#     else:
#         pass
# import copy
#
# reload_state_dict = {}
# for k, v in master_model.state_dict().items():
#     parameter_type = k.split('.')[-1]
#     if 'weight' in parameter_type or 'bias' in parameter_type:
#         if 'weight' in parameter_type:
#             if v.dim() > 1:
#                 reload_state_dict[k] = copy.deepcopy(v[torch.meshgrid(param_idx[k])])
#             else:
#                 reload_state_dict[k] = copy.deepcopy(v[param_idx[k]])
#         else:
#             reload_state_dict[k] = copy.deepcopy(v[param_idx[k]])
#     else:
#         reload_state_dict[k] = copy.deepcopy(v)
# client_model.load_state_dict(reload_state_dict)