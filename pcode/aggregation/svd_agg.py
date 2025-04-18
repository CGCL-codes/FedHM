from collections import OrderedDict
from pcode.utils.module_state import ModuleState
import torch
import copy
import re

attn_weight_pattern = '.*attn\.\w+\.weight'
def spectral_init(weight, rank):
    U, S, V = torch.svd(weight)
    sqrtS = torch.diag(torch.sqrt(S[:rank]))

    u_weight_sliced, v_weight_sliced = torch.matmul(U[:, :rank], sqrtS), torch.matmul(
        V[:, :rank], sqrtS).T

    return u_weight_sliced, v_weight_sliced


class SVDAggregator:
    def __init__(self, conf, master_model, client_models, label_split):
        self.initial_weight = None
        self.conf = conf
        self.used_client_archs = self.conf.used_client_archs
        self.master_model, self.client_models = master_model, client_models
        self.clientid2arch = self.conf.clientid2arch
        self.label_split = label_split

        if 'vit' not in self.conf.arch_info["master"]:
            self.init_conv_index()
        else:
            self.init_transformer(master_model)

    def init_transformer(self, master_model):
        self.initial_weight = ModuleState(copy.deepcopy(master_model.state_dict()))

    def init_conv_index(self):
        self.index_map = dict((arch,
                               OrderedDict()
                               ) for arch in self.used_client_archs)

        for arch in self.used_client_archs:
            rank_factor = eval(arch.split('_')[-1])
            if rank_factor <= 1:
                continue

            decompose_name, skip_name, upper_bound = self.decide_upper_bound(rank_factor)
            master_conv_index, master_other_indx = [], []
            for index, (param_name, param) in enumerate(self.master_model.state_dict().items()):
                if 'conv' in param_name and skip_name not in param_name \
                        and index not in range(0, upper_bound):
                    master_conv_index.append(index)
                else:
                    master_other_indx.append(index)

            local_conv_indx, local_other_indx = [], []
            for index, (param_name, param) in enumerate(self.client_models[arch].state_dict().items()):
                if 'bn1_u' in param_name or 'bn2_u' in param_name:
                    continue
                if 'conv' in param_name and index not in range(0, upper_bound) and skip_name not in param_name:
                    local_conv_indx.append(index)
                else:
                    local_other_indx.append(index)
            local_index = 0
            for index in master_conv_index:
                self.index_map[arch][index] = local_conv_indx[local_index]
                local_index += 2

            local_index = 0
            for index in master_other_indx:
                self.index_map[arch][index] = local_other_indx[local_index]
                local_index += 1
            return

    def split_conv(self, master_state, rank_factor, arch, client_models):
        reconstructed_aggregator = copy.deepcopy(list(client_models[arch].state_dict().values()))
        decompose_name, skip_name, upper_bound = self.decide_upper_bound(rank_factor)

        for index, (param_name, param) in enumerate(master_state.items()):

            if 'conv' in param_name and index not in range(0, upper_bound) \
                    and skip_name not in param_name:
                out_channels, in_channels, ks1, ks2 = param.shape
                dim1, dim2 = out_channels * ks1, in_channels * ks2
                param_reshaped = param.permute(0, 2, 1, 3).reshape(dim1, dim2)
                # param_reshaped = param.view(dim1, dim2)

                if decompose_name not in param_name:
                    sliced_rank = out_channels
                else:
                    sliced_rank = max(int(round(out_channels / rank_factor)), 1)

                u_weight_sliced, v_weight_sliced = spectral_init(param_reshaped, sliced_rank)

                reconstructed_aggregator[self.index_map[arch][index]] = v_weight_sliced
                reconstructed_aggregator[self.index_map[arch][index] + 1] = u_weight_sliced
            else:
                reconstructed_aggregator[self.index_map[arch][index]] = param

        reload_state_dict = {}

        for item_index, (param_name, param) in enumerate(client_models[arch].state_dict().items()):
            reload_state_dict[param_name] = param
            if 'bn1_u' in param_name or 'bn2_u' in param_name:
                continue
            reload_state_dict[param_name] = reconstructed_aggregator[item_index]

        return reload_state_dict

    def split_transformer(self, master_state, rank_factor, arch, client_models):
        reload_state_dict = copy.deepcopy(master_state)

        # calculate the diff of update
        master_state = ModuleState(master_state)
        state_diff = master_state - self.initial_weight

        for param_name, param in state_diff.state_dict.items():
            if re.match(attn_weight_pattern, param_name) is None:
                continue
            prefix_name = param_name.rsplit('.', 1)[0]

            U, VT = spectral_init(param, rank_factor)
            reload_state_dict[prefix_name + '.lora_A'] = VT
            reload_state_dict[prefix_name + '.lora_B'] = U

        return reload_state_dict

    def split_model(self, master_model, client_models):
        # accelarate computing
        with torch.no_grad():
            master_model = master_model.cuda()
            master_state = copy.deepcopy(master_model.state_dict())

            for arch in self.used_client_archs:
                rank_factor = 1

                if len(arch.split('_')) > 1:
                    rank_factor = eval(arch.split('_')[-1])

                if rank_factor > 1:
                    if 'vit' in arch:
                        client_models[arch].train()
                        reload_state_dict = self.split_transformer(master_state, rank_factor, arch, client_models)
                    else:
                        reload_state_dict = self.split_conv(master_state, rank_factor, arch, client_models)
                else:
                    reload_state_dict = master_state

                client_models[arch].load_state_dict(reload_state_dict)
                client_models[arch] = client_models[arch].cpu()
        master_model = master_model.cpu()
        return client_models

    def aggregate_conv(self, global_params, local_model_params, rank_factor, _arch, reload_state_dict, weights,
                       client_idx):

        label_split = self.label_split[client_idx]
        decompose_name, skip_name, upper_bound = self.decide_upper_bound(rank_factor)
        for index, (param_name, param) in enumerate(global_params):
            if 'conv' in param_name \
                    and index not in range(0, upper_bound) \
                    and skip_name not in param_name:

                # fusion.fuse_conv_bn_weights()
                model_u_weight, model_v_weight = local_model_params[self.index_map[_arch][index]], \
                                                 local_model_params[self.index_map[_arch][index] + 1]

                u_weight = model_v_weight  # .view(dim1, rank)
                v_weight = model_u_weight  # .view(rank, dim2)

                combine_weights = torch.matmul(u_weight, v_weight)
                combine_weights = combine_weights.reshape(param.shape[0], param.shape[2], param.shape[1],
                                                          param.shape[3]).permute(0, 2, 1, 3)
                # combine_weights = combine_weights.view(param.shape)

                assert combine_weights.size() == param.data.size()
                reload_state_dict[param_name] += weights[client_idx] * combine_weights
            elif "classifier" in param_name:
                reload_state_dict[param_name][label_split] += weights[client_idx] * \
                                                              local_model_params[self.index_map[_arch][index]][
                                                                  label_split]
            else:
                reload_state_dict[param_name] = reload_state_dict[param_name] + \
                                                weights[client_idx] * local_model_params[
                                                    self.index_map[_arch][index]]
        return reload_state_dict

    def aggregate_transformer(self, local_model_state, _arch, client_idx, weights, reload_state_dict):
        for param_name, param in local_model_state.items():
            if 'lora_' in param_name:
                continue
            reload_state_dict[param_name] += weights[client_idx] * local_model_state[param_name]
        return reload_state_dict

    def aggregate_model(self, flatten_local_models):
        with torch.no_grad():
            self.master_model = self.master_model.cuda()
            global_params = list(self.master_model.state_dict().items())
            reload_state_dict = {}
            for (param_name, param) in global_params:
                reload_state_dict[param_name] = torch.zeros_like(param.data)

            # weight = 1.0 / float(len(flatten_local_models))
            local_models = {}
            factors_num = set()
            for client_idx, flatten_local_model in flatten_local_models.items():
                _arch = self.clientid2arch[client_idx]

                rank_factor = 1
                if len(_arch.split('_')) > 1:
                    rank_factor = eval(_arch.split('_')[-1])

                factors_num.add(rank_factor)

                _model = copy.deepcopy(self.client_models[_arch])
                _model_state_dict = self.client_models[_arch].state_dict()
                flatten_local_model.unpack(_model_state_dict.values())
                _model.load_state_dict(_model_state_dict)
                _model = _model.eval()  # for lora weights

                local_models[client_idx] = (_model, rank_factor, _arch)

            factors_num = list(factors_num)
            factors_num.sort(reverse=True)

            # preserve higher dimension history info
            min_rank_factor = int(self.conf.arch_info["worker"][0].split('_')[-1])
            if factors_num[-1] > min_rank_factor:
                local_models[0] = (copy.deepcopy(self.master_model), 1, self.conf.arch_info["worker"][0])

            weights = self.get_client_model_weights(local_models)

            for client_idx, (_model, rank_factor, _arch) in local_models.items():
                _model = _model.cuda()
                local_model_params = list(_model.state_dict().values())
                if rank_factor > 1:
                    if 'vit' in self.conf.arch_info["master"]:
                        reload_state_dict = self.aggregate_transformer(_model.state_dict(), _arch, client_idx, weights,
                                                                       reload_state_dict)
                    else:
                        reload_state_dict = self.aggregate_conv(global_params, local_model_params, rank_factor,
                                                                _arch, reload_state_dict, weights, client_idx)
                else:
                    for index, (param_name, param) in enumerate(global_params):
                        reload_state_dict[param_name] = reload_state_dict[param_name] + weights[client_idx] * \
                                                        local_model_params[index]

            self.master_model.load_state_dict(reload_state_dict)
            return self.master_model

    def get_client_model_weights(self, local_models):
        if not self.conf.dynamic:
            return dict((k, float(1.0 / len(local_models))) for k in local_models.keys())

        weights_values = []
        index_map = {}

        for index, (client_idx, (_model, rank_factor, _arch)) in enumerate(local_models.items()):
            weights_values.append(-rank_factor)
            index_map[client_idx] = index

        temperature = self.conf.softmax_temperature

        weights_values = torch.softmax(torch.tensor(weights_values) / temperature, dim=0)
        weights = {}
        for client_idx in local_models.keys():
            weights[client_idx] = weights_values[index_map[client_idx]]

        return weights

    def decide_upper_bound(self, rank_factor):

        scaler = 1 if self.conf.freeze_bn else 2
        if 'resnet50' in self.conf.arch_info["master"]:
            upper_bound = 33 * scaler
            skip_name = 'downsample'
            decompose_name = 'conv'
        elif 'resnet18' in self.conf.arch_info["master"]:
            upper_bound = 8 * scaler
            skip_name = 'downsample'
            decompose_name = 'conv'
        elif 'resnet34' in self.conf.arch_info["master"]:
            upper_bound = 48 * scaler
            skip_name = 'downsample'
            decompose_name = 'conv'
        else:
            upper_bound = 1
            skip_name = 'transition'
            decompose_name = 'conv1'

        if rank_factor > 64:
            upper_bound = 1 * scaler

        return decompose_name, skip_name, upper_bound
