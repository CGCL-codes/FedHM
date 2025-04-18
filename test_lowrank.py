from torchvision.models import resnet18
from pcode.models.lowrank_resnet import HybridResNet, LowRankBasicBlockConv1x1,LowRankBasicBlock,FullRankBasicBlock
from pcode.models.resnet import ResNet_imagenet,FullRankBasicBlock, PruningBasicBlock
from torch.nn.utils import fuse_conv_bn_weights
import torch

def param_counter(model):
    num_params = 0
    for param_index, (param_name, param) in enumerate(model.named_parameters()):
        num_params += param.numel()
    return num_params / 1e6
torch.manual_seed(42)

# fullrank_model = ResNet_imagenet("cifar10",18,freeze_bn=True)
# for index , param_name in enumerate(fullrank_model.state_dict().keys()):
#     print(index, param_name)
# model_params = {
#     18: {"block": LowRankBasicBlockConv1x1, "layers": [2, 2, 2, 2]},
#     34: {"block": LowRankBasicBlockConv1x1, "layers": [3, 4, 6, 3]},
#     50: {"block": LowRankBasicBlockConv1x1, "layers": [3, 4, 6, 3]},
#     101: {"block": LowRankBasicBlockConv1x1, "layers": [3, 4, 23, 3]},
#     152: {"block": LowRankBasicBlockConv1x1, "layers": [3, 8, 36, 3]},
# }
# block = model_params[18]["block"]
# layers = model_params[18]["layers"]
# rank_factor = 4

# lowrank_model = HybridResNet(block,FullRankBasicBlock, rank_factor, layers,
#                                      num_classes=10,track_running_stats=False)
#
# model_params = list(fullrank_model.state_dict().items())
#
# local_model_params = list(lowrank_model.state_dict().values())
# print(lowrank_model.state_dict().keys())
#
#
# print("@@@ Num Params: Vanilla Model: {}, Hybrid Model: {}".format(param_counter(fullrank_model),
#                                                                          param_counter(lowrank_model)))

from collections import OrderedDict
def init_svd(fullrank_model, lowrank_model, arch):
    with torch.no_grad():

        global_model_params = fullrank_model.state_dict().items()
        lowrank_model_params = lowrank_model.state_dict().items()

        if '50' in arch:
            upper_bound = 33
            skip_name = 'downsample'
        elif '18' in arch:
            upper_bound = 8 * 2
            skip_name = 'downsample'
        elif '34' in arch:
            upper_bound = 48 * 2
            skip_name = 'downsample'
        elif 'efficient' in arch:
            upper_bound = 176
            skip_name = 'dpwise'
        else:
            #upper_bound = 199
            upper_bound = 1
            skip_name = 'transition'

        conv_index, other_indx = [], []
        for index, (param_name, param) in enumerate(global_model_params):
            if 'conv' in param_name and skip_name not in param_name\
                    and index not in range(0, upper_bound):
                conv_index.append(index)
            else:
                other_indx.append(index)

        local_conv_indx, local_other_indx = [], []

        for index, (param_name, param) in enumerate(lowrank_model_params):
            print(index, param_name)
            if 'bn1_u' in param_name or 'bn2_u' in param_name:
                continue
            if 'conv' in param_name and index not in range(0, upper_bound) \
                and skip_name not in param_name:
                local_conv_indx.append(index)
            else:
                local_other_indx.append(index)
        index_map = {arch: OrderedDict()}
        local_index = 0
        for index in conv_index:
            index_map[arch][index] = local_conv_indx[local_index]
            local_index += 2
        local_index = 0
        for index in other_indx:
            index_map[arch][index] = local_other_indx[local_index]
            local_index += 1

        return index_map,arch

import copy
def svd_split_model(fullrank_model,lowrank_model,index_map,arch, bn_bias, square = False):
    with torch.no_grad():
        #fullrank_model.eval()
        #copied_model = copy.deepcopy(lowrank_model)
        rank_factor = int(arch.split('_')[1])
        #resnet_size = int((arch.split('_')[0].replace("resnet", "")))
        if '50' in arch:
            upper_bound = 33
            skip_name = 'downsample'
        elif '18' in arch:
            upper_bound = 8 * 2
            skip_name = 'downsample'
        elif '34' in arch:
            upper_bound = 48 * 2
            skip_name = 'downsample'
        elif 'efficient' in arch:
            upper_bound = 176
            skip_name = 'dpwise'
        else:
            #upper_bound = 199
            upper_bound = 1
            skip_name = 'transition'

        local_model_params = list(lowrank_model.state_dict().values())
        reconstruct_aggregator = [0] * len(local_model_params)
        model_params = list(fullrank_model.state_dict().items())

        for index, (param_name, param) in enumerate(fullrank_model.state_dict().items()):
            if index < upper_bound  or 'conv' not in param_name or skip_name in param_name:
                param_name, param = model_params[index]
                reconstruct_aggregator[index_map[arch][index]] = param
            else:
                out_channels, in_channels, ks1, ks2 = param.shape
                # if square:
                #     dim1, dim2 = out_channels, in_channels * ks2 * ks1
                #     param_reshaped = param.view(dim1, dim2)
                dim1, dim2 = out_channels * ks1, in_channels * ks2
                param_reshaped = param.permute(0, 2, 1, 3).reshape(dim1, dim2)
                sliced_rank = max(int(round(out_channels / rank_factor)) , 1)

                U, S, V = torch.svd(param_reshaped)
                sqrtS = torch.diag(torch.sqrt(S[:sliced_rank]))

                u_weight_sliced,v_weight_sliced = torch.matmul(U[:,:sliced_rank], sqrtS), torch.matmul(V[:,:sliced_rank], sqrtS).T

                model_weight_v = u_weight_sliced#.reshape(out_channels, ks1, sliced_rank, 1).permute(0, 2, 1, 3)
                model_weight_u = v_weight_sliced#.T.reshape(in_channels, ks2, 1, sliced_rank).permute(3, 0, 2, 1)
                #print(torch.matmul(u_weight_sliced,v_weight_sliced)[:10,:10])

                #print(param_reshaped[:10,:10])
                reconstruct_aggregator[index_map[arch][index]] = model_weight_u
                reconstruct_aggregator[index_map[arch][index] + 1] = model_weight_v

        reload_state_dict = {}

        for item_index, (param_name, param) in enumerate(lowrank_model.state_dict().items()):
            if 'bn1_u' in param_name or 'bn2_u' in param_name:
                continue
            reload_state_dict[param_name] = reconstruct_aggregator[item_index]

        lowrank_model.load_state_dict(reload_state_dict)
        return lowrank_model

from  torch.nn.utils.fusion import fuse_conv_bn_weights

def solve_svd(A,b):
    # compute svd of A
    U, s, Vh = torch.svd(A)

    # U diag(s) Vh x = b <=> diag(s) Vh x = U.T b = c
    c = torch.matmul(U.T, b)
    # diag(s) Vh x = c <=> Vh x = diag(1/s) c = w (trivial inversion of a diagonal matrix)
    w = torch.matmul(torch.diag(1 / s), c)
    # Vh x = w <=> x = Vh.H w (where .H stands for hermitian = conjugate transpose)
    x = torch.matmul(Vh.conj().T, w)
    return x

def svd_combine_model(fullrank_model,lowrank_model,index_map,arch, square = True):
    with torch.no_grad():

        rank_factor = eval(arch.split('_')[-1])
        upper_bound = 33
        local_model_params_dict = lowrank_model.state_dict()
        local_model_params = list(lowrank_model.state_dict().values())
        reload_state_dict = {}
        full_ratio, low_ratio = 0, 1
        global_param_dict = fullrank_model.state_dict()
        #test_model = ResNet_imagenet("tiny-imagenet",18,freeze_bn=False).cuda()
        #bn_bias = {}
        for index, (param_name, param) in enumerate(global_param_dict.items()):
            if index < upper_bound or 'downsample' in param_name or len(param.shape) != 4:
                param.data = (local_model_params[index_map[arch][index]] * low_ratio + param.data *full_ratio )
                reload_state_dict[param_name] = param.data
            else:
                # if "conv1" in param_name:
                #     bn_u_name = ".".join([param_name.split(".")[:2],"bn1_u"])
                # else:
                #     bn_u_name = ".".join([param_name.split(".")[:2], "bn2_u"])

                model_u_weight, model_v_weight = local_model_params[index_map[arch][index]], \
                                                 local_model_params[index_map[arch][index] + 1]

                # bn_weight,bn_bias,bn_rm, bn_rv = local_model_params_dict[bn_u_name+".weight"],local_model_params_dict[bn_u_name+".bias"],\
                #                                 local_model_params_dict[bn_u_name + ".running_mean"], local_model_params_dict[bn_u_name + ".running_var"]

                # model_u_weight,model_u_bias = fuse_conv_bn_weights(model_u_weight,None,bn_rm, bn_rv,1e-5,bn_weight,bn_bias)

                u_weight = model_v_weight#.view(dim1, rank)
                v_weight = model_u_weight#.view(rank, dim2)

                combine_weights = torch.matmul(u_weight, v_weight)
                #combine_bias = torch.matmul(model_u_bias.unsqueeze(0), model_v_weight.T).squeeze()

                combine_weights = combine_weights.reshape(param.shape[0], param.shape[2], param.shape[1], param.shape[3]).permute(0, 2, 1, 3)
                #combine_weights = combine_weights.view(param.shape)
                assert combine_weights.size() == param.data.size()
                reload_state_dict[param_name] = (combine_weights * low_ratio + param.data * full_ratio )

                #conv.bias -> add to bn bias
                # if "conv1" in param_name:
                #     bias_name = ".".join([param_name.split(".")[:2], "bn1.bias"])
                # else:
                #     bias_name = ".".join([param_name.split(".")[:2], "bn2.bias"])
                # reload_state_dict[bias_name] += combine_bias * low_ratio
                # bn_bias[bias_name] = combine_bias * low_ratio

        fullrank_model.load_state_dict(reload_state_dict)
        return fullrank_model

import numpy as np
import math
def init_pruning_index(master_model, arch = 'resnet18_1/2',compression=0.5):
    idx_i = {arch: None}
    param_idx = {arch: OrderedDict()}

    blk_count = 0
    last_channels , num_channels = 0 , 0
    for k, v in master_model.state_dict().items():
        parameter_type = k.split('.')[-1]
        scaler_rate = eval(arch.split('_')[-1])
        if 'weight' in parameter_type or 'bias' in parameter_type:
            if parameter_type == 'weight':
                if v.dim() > 1:
                    input_size = v.size(1)
                    output_size = v.size(0)
                    if 'conv' in k:
                        if idx_i[arch] is None:
                            idx_i[arch] = torch.arange(input_size, device=v.device)
                        input_idx_i_m = idx_i[arch]
                        # scaler_rate = self.model_rate[archs[m]]
                        local_output_size = int(round(output_size * scaler_rate))
                        output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                        idx_i[arch] = output_idx_i_m
                    elif 'downsample.0' in k:
                        input_idx_i_m = param_idx[arch][k.replace('downsample.0', 'conv1')][1]
                        output_idx_i_m = idx_i[arch]

                    elif 'classifier' in k:
                        input_idx_i_m = idx_i[arch]
                        output_idx_i_m = torch.arange(output_size, device=v.device)
                    else:
                        raise ValueError('Not valid k')
                    param_idx[arch][k] = (output_idx_i_m, input_idx_i_m)
                else:
                    input_idx_i_m = idx_i[arch]
                    param_idx[arch][k] = input_idx_i_m
            else:
                input_size = v.size(0)
                if 'classifier' in k:
                    input_idx_i_m = torch.arange(input_size, device=v.device)
                    param_idx[arch][k] = input_idx_i_m
                else:
                    input_idx_i_m = idx_i[arch]
                    param_idx[arch][k] = input_idx_i_m
        elif 'mean' in k or 'var' in k:
            input_idx_i_m = idx_i[arch]
            param_idx[arch][k] = input_idx_i_m
        # if 'weight' in parameter_type or 'bias' in parameter_type:
        #     if parameter_type == 'weight':
        #         if v.dim() > 1:
        #             input_size = v.size(1)
        #             output_size = v.size(0)
        #             if 'transition' in k:
        #
        #                 input_idx_i_m = torch.arange(num_channels,device=v.device)
        #                 num_channels = int(math.floor(num_channels * compression))
        #                 output_idx_i_m = torch.arange(num_channels, device=v.device)
        #                 last_channels = num_channels
        #                 blk_count += 1
        #             elif 'conv' in k :
        #                 local_output_size = int(round(output_size * scaler_rate))
        #                 if blk_count == 0:
        #                     input_idx_i_m = torch.arange(input_size,device=v.device)
        #                     num_channels = local_output_size
        #                     last_channels = num_channels
        #                     blk_count += 1
        #                 elif 'conv1' in k:
        #                     input_idx_i_m = torch.arange(last_channels, device=v.device)
        #                     last_channels = local_output_size
        #                 else:
        #                     input_idx_i_m = torch.arange(last_channels,device=v.device)
        #                     num_channels += local_output_size
        #                     last_channels = num_channels
        #
        #                 output_idx_i_m = torch.arange(local_output_size, device=v.device)
        #             elif 'downsample.0' in k:
        #                 input_idx_i_m = param_idx[arch][k.replace('downsample.0', 'conv1')][1]
        #                 local_output_size = int(round(output_size * scaler_rate))
        #                 output_idx_i_m = torch.arange(local_output_size, device=v.device)
        #                 last_channels = local_output_size
        #             elif 'classifier' in k:
        #                 input_idx_i_m =  torch.arange(last_channels, device=v.device)
        #                 output_idx_i_m = torch.arange(output_size, device=v.device)
        #             else:
        #                 raise ValueError('Not valid k')
        #             param_idx[arch][k] = (output_idx_i_m, input_idx_i_m)
        #         else:
        #             input_idx_i_m = torch.arange(last_channels, device=v.device)
        #             param_idx[arch][k] = input_idx_i_m
        #     else:
        #         input_size = v.size(0)
        #         if 'classifier' in k:
        #             input_idx_i_m = torch.arange(input_size, device=v.device)
        #             param_idx[arch][k] = input_idx_i_m
        #         else:
        #             input_idx_i_m = torch.arange(last_channels, device=v.device)
        #             param_idx[arch][k] = input_idx_i_m
        # elif 'mean' in k or 'var' in k:
        #     input_idx_i_m = torch.arange(last_channels, device=v.device)
        #     param_idx[arch][k] = input_idx_i_m

    return param_idx,arch

def pruning_split_model(param_idx, arch, client_model,master_model):
    with torch.no_grad():
        assert param_idx is not None

        reload_state_dict = {}
        for k, v in master_model.state_dict().items():
            parameter_type = k.split('.')[-1]
            if 'weight' in parameter_type or 'bias' in parameter_type:
                if 'weight' in parameter_type:
                    if v.dim() > 1:
                        reload_state_dict[k] = copy.deepcopy(v[torch.meshgrid(param_idx[arch][k])])
                    else:
                        reload_state_dict[k] = copy.deepcopy(v[param_idx[arch][k]])
                else:
                    reload_state_dict[k] = copy.deepcopy(v[param_idx[arch][k]])
            elif 'mean' in parameter_type or  'var' in parameter_type:
                reload_state_dict[k] = copy.deepcopy(v[param_idx[arch][k]])
            else:
                reload_state_dict[k] = copy.deepcopy(v)

        client_model.load_state_dict(reload_state_dict)
    return client_model

import torch.nn.functional as F
from tqdm import tqdm
def _divergence(student_logits, teacher_logits):
    divergence =  F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.softmax(teacher_logits, dim=1),
        reduction="batchmean",
    )  # forward KL
    return divergence

def distill_large_model(master_model, split_model, distill_loader):
    optimizer = torch.optim.Adam(split_model.parameters(),lr = 1e-3)
    master_model.eval()
    split_model.train()
    split_model.requires_grad_(True)
    step,total_step = 0, 100
    for input, target in tqdm(distill_loader):
        input, target = input.cuda(), target.cuda()
        _, teacher_output = master_model(input)
        _, student_output = split_model(input)

        optimizer.zero_grad()
        loss = _divergence(student_output,teacher_output.detach())
        loss.backward()
        optimizer.step()
        step += 1
        if step > total_step:
            break

    return split_model

# from pcode.models.lowrank_resnet import HybridResNet, LowRankBasicBlockConv1x1,FullRankBasicBlock,LowRankBasicBlock,LowRankBottleneckConv1x1,Bottleneck
# rank_factor = 4
#
# lowrank_model = HybridResNet(LowRankBottleneckConv1x1, Bottleneck,rank_factor=rank_factor, layers=[3, 4, 6, 3],
#                                   num_classes=10,track_running_stats=False).cuda()
# fullrank_model = ResNet_imagenet("cifar10",50,freeze_bn=True).cuda()
#
# index_map, arch = init_svd(fullrank_model,lowrank_model,'resnet50_4')
# lowrank_model = svd_split_model(fullrank_model, lowrank_model, index_map, arch,False)


# data = torch.randn(1, 3, 32, 32)
# y = lowrank_model(data)

# from parameters import get_args
# from pcode.aggregation.svd_agg import Aggregator
# from pcode.create_model import define_model,determine_arch
# from pcode.utils.tensor_buffer import TensorBuffer
# import pcode.utils.logging as logging
# import pcode.utils.checkpoint as checkpoint
# import pcode.utils.param_parser as param_parser
# import pcode.utils.topology as topology
#
# conf = get_args()
# conf.timestamp='0'
# conf.graph = topology.define_graph_topology(
#         world=conf.world,
#         world_conf=conf.world_conf,
#         n_participated=conf.n_participated,
#         on_cuda=conf.on_cuda,
#     )
# conf.graph.rank = 0
#
# conf._fl_aggregate = conf.fl_aggregate
# conf.fl_aggregate = (
#     param_parser.dict_parser(conf.fl_aggregate)
#     if conf.fl_aggregate is not None
#     else conf.fl_aggregate
# )
# [setattr(conf, f"fl_aggregate_{k}", v) for k, v in conf.fl_aggregate.items()]
# checkpoint.init_checkpoint(conf, rank=str(0))
# conf.logger = logging.Logger(conf.checkpoint_dir)
# conf.arch_info = (
#         param_parser.dict_parser(conf.complex_arch)
#         if conf.complex_arch is not None
#         else {"master": conf.arch, "worker": conf.arch}
#     )
# conf.arch_info["worker"] = conf.arch_info["worker"].split(":")
# conf.used_client_archs = conf.arch_info["worker"]
#
# _,master_model = define_model(conf,to_consistent_model=False,arch='resnet18_1',on_cuda=False)
#
# worker_archs = ['1','2','4','18']
# client_models = dict(
#             define_model(conf, to_consistent_model=False, arch=str('resnet18_'+ i),on_cuda=False)
#             for i in worker_archs
# )
#
# svd_agg = Aggregator(conf,master_model,client_models)
#
# clientid2arch = dict(
#             (
#                 client_id,
#                 determine_arch(
#                     conf, client_id=client_id, use_complex_arch=True
#                 ),
#             )
#             for client_id in range(1, 1 + conf.n_clients)
#         )
#
# flatten_local_models = dict()
# for client_id in range(2,1+conf.n_clients):
#     arch = clientid2arch[client_id]
#     client_tb = TensorBuffer(
#         list(client_models[arch].state_dict().values())
#     )
#     client_tb.buffer = torch.zeros_like(client_tb.buffer)
#     flatten_local_models[client_id] = client_tb
#
# import  time
# start = time.time()
# svd_agg.svd_combine_by_unit(clientid2arch,master_model,flatten_local_models,client_models)
# elapsed_time = time.time() - start
# print(elapsed_time)