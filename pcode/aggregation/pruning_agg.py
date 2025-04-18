from collections import OrderedDict
import torch
import copy
import math

class HeteroAggregator():
    def __init__(self, conf, master_model, client_models, label_split):
        self.conf = conf
        self.used_client_archs = self.conf.used_client_archs
        self.master_model, self.client_models = master_model, client_models
        self.clientid2arch = self.conf.clientid2arch
        self.label_split = label_split

        self.init_index()

    def init_index(self):
        with torch.no_grad():
            idx_i = dict((arch, None)
                         for arch in self.used_client_archs)
            self.param_idx = dict((arch,
                                   OrderedDict()
                                   ) for arch in self.used_client_archs)
            blk_count = 0
            for k, v in self.master_model.state_dict().items():
                parameter_type = k.split('.')[-1]
                for arch in self.used_client_archs:
                    scaler_rate = eval(arch.split('_')[-1])
                    if "resnet" in self.conf.arch_info["master"]:
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
                                        input_idx_i_m = self.param_idx[arch][k.replace('downsample.0', 'conv1')][1]
                                        output_idx_i_m = idx_i[arch]
                                    elif 'classifier' in k:
                                        input_idx_i_m = idx_i[arch]
                                        output_idx_i_m = torch.arange(output_size, device=v.device)
                                    else:
                                        raise ValueError('Not valid k')
                                    self.param_idx[arch][k] = (output_idx_i_m, input_idx_i_m)
                                else:
                                    input_idx_i_m = idx_i[arch]
                                    self.param_idx[arch][k] = input_idx_i_m
                            else:
                                input_size = v.size(0)
                                if 'classifier' in k:
                                    input_idx_i_m = torch.arange(input_size, device=v.device)
                                    self.param_idx[arch][k] = input_idx_i_m
                                else:
                                    input_idx_i_m = idx_i[arch]
                                    self.param_idx[arch][k] = input_idx_i_m
                        elif 'mean' in k or 'var' in k:
                            input_idx_i_m = idx_i[arch]
                            self.param_idx[arch][k] = input_idx_i_m

                    else:  # densenet
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            if parameter_type == 'weight':
                                if v.dim() > 1:
                                    input_size = v.size(1)
                                    output_size = v.size(0)
                                    if 'transition' in k:
                                        input_idx_i_m = torch.arange(num_channels, device=v.device)
                                        num_channels = int(math.floor(num_channels * self.conf.densenet_compression))
                                        output_idx_i_m = torch.arange(num_channels, device=v.device)
                                        last_channels = num_channels
                                        blk_count += 1
                                    elif 'conv' in k:
                                        local_output_size = int(round(output_size * scaler_rate))
                                        if blk_count == 0:
                                            input_idx_i_m = torch.arange(input_size, device=v.device)
                                            num_channels = local_output_size
                                            last_channels = num_channels
                                            blk_count += 1
                                        elif 'conv1' in k:
                                            input_idx_i_m = torch.arange(last_channels, device=v.device)
                                            last_channels = local_output_size
                                        else:
                                            input_idx_i_m = torch.arange(last_channels, device=v.device)
                                            num_channels += local_output_size
                                            last_channels = num_channels

                                        output_idx_i_m = torch.arange(local_output_size, device=v.device)
                                    elif 'downsample.0' in k:
                                        input_idx_i_m = self.param_idx[arch][k.replace('downsample.0', 'conv1')][1]
                                        local_output_size = int(round(output_size * scaler_rate))
                                        output_idx_i_m = torch.arange(local_output_size, device=v.device)
                                        last_channels = local_output_size
                                    elif 'classifier' in k:
                                        input_idx_i_m = torch.arange(last_channels, device=v.device)
                                        output_idx_i_m = torch.arange(output_size, device=v.device)
                                    else:
                                        raise ValueError('Not valid k')
                                    self.param_idx[arch][k] = (output_idx_i_m, input_idx_i_m)
                                else:
                                    input_idx_i_m = torch.arange(last_channels, device=v.device)
                                    self.param_idx[arch][k] = input_idx_i_m
                            else:
                                input_size = v.size(0)
                                if 'classifier' in k:
                                    input_idx_i_m = torch.arange(input_size, device=v.device)
                                    self.param_idx[arch][k] = input_idx_i_m
                                else:
                                    input_idx_i_m = torch.arange(last_channels, device=v.device)
                                    self.param_idx[arch][k] = input_idx_i_m
                        elif 'mean' in k or 'var' in k:
                            input_idx_i_m = torch.arange(last_channels, device=v.device)
                            self.param_idx[arch][k] = input_idx_i_m

    def split_model(self, master_model, client_models):
        with torch.no_grad():
            assert self.param_idx is not None
            master_model = master_model.cuda()
            for arch in self.used_client_archs:
                reload_state_dict = {}
                for k, v in self.master_model.state_dict().items():
                    parameter_type = k.split('.')[-1]
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if 'weight' in parameter_type:
                            if v.dim() > 1:
                                reload_state_dict[k] = copy.deepcopy(v[torch.meshgrid(self.param_idx[arch][k])])
                            else:
                                reload_state_dict[k] = copy.deepcopy(v[self.param_idx[arch][k]])
                        else:
                            reload_state_dict[k] = copy.deepcopy(v[self.param_idx[arch][k]])
                    elif 'mean' in parameter_type or 'var' in parameter_type:
                        reload_state_dict[k] = copy.deepcopy(v[self.param_idx[arch][k]])
                    else:
                        pass
                        # reload_state_dict[k] = copy.deepcopy(v)

                client_models[arch].load_state_dict(reload_state_dict)
                client_models[arch] = client_models[arch].cpu()
            master_model = master_model.cpu()
        return client_models

    def aggregate_model(self, flatten_local_models):
        with torch.no_grad():
            count = OrderedDict()
            self.master_model = self.master_model.cuda()
            global_params = self.master_model.state_dict()
            local_client_parameters = {}
            # client tensor -> arch + state_dict
            for client_idx, flatten_local_model in flatten_local_models.items():
                _arch = self.clientid2arch[client_idx]
                _model_state_dict = self.client_models[_arch].state_dict()
                flatten_local_model.unpack(_model_state_dict.values())
                label_split = torch.tensor(self.label_split[client_idx])
                local_client_parameters[client_idx] = (_arch, _model_state_dict, label_split)

            # client state_dict -> avg
            reload_state_dict = {}
            for k, v in global_params.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)

                for client_idx, client_parameters in local_client_parameters.items():
                    _arch, local_parameters, label_split = client_parameters
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                if 'classifier' in k:
                                    param_idx = list(copy.deepcopy(self.param_idx[_arch][k]))
                                    param_idx[0] = param_idx[0][label_split]

                                    tmp_v[torch.meshgrid(param_idx)] += local_parameters[k][label_split]
                                    count[k][torch.meshgrid(param_idx)] += 1
                                else:
                                    tmp_v[torch.meshgrid(self.param_idx[_arch][k])] += local_parameters[k]
                                    count[k][torch.meshgrid(self.param_idx[_arch][k])] += 1
                            else:
                                tmp_v[self.param_idx[_arch][k]] += local_parameters[k]
                                count[k][self.param_idx[_arch][k]] += 1
                        else:
                            if 'classifier' in k:
                                param_idx = self.param_idx[_arch][k][label_split]
                                tmp_v[param_idx] += local_parameters[k][label_split]
                                count[k][param_idx] += 1
                            else:
                                tmp_v[self.param_idx[_arch][k]] += local_parameters[k]
                                count[k][self.param_idx[_arch][k]] += 1
                    elif 'mean' in parameter_type or 'var' in parameter_type:
                        tmp_v[self.param_idx[_arch][k]] += local_parameters[k]
                        count[k][self.param_idx[_arch][k]] += 1
                    else:
                        pass

                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
                reload_state_dict[k] = copy.deepcopy(v)

            self.master_model.load_state_dict(reload_state_dict)
            return self.master_model

    def get_client_model_weights(self, local_models):
        # for non i.i.d situation, change the weights
        if self.conf.partition_data == "non_iid_dirichlet" or not self.conf.dynamic:
            return dict((k, float(1.0 / len(local_models))) for k in local_models.keys())

        weights_values = []
        index_map = {}

        for index, (client_idx, (_model, rank_factor, _arch)) in enumerate(local_models.items()):
            weights_values.append(-rank_factor)
            index_map[client_idx] = index

        temperature = self.conf.loss_temperature

        milstones = [int(x) for x in self.conf.lr_milestones.split(",")]
        if self.conf.graph.comm_round > milstones[0]:
            temperature = self.conf.loss_temperature * 2

        weights_values = torch.softmax(torch.tensor(weights_values) / temperature, dim=0)
        weights = {}
        for client_idx in local_models.keys():
            weights[client_idx] = weights_values[index_map[client_idx]]

        # print(weights)
        return weights





