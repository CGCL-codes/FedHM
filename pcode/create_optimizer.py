# -*- coding: utf-8 -*-
import torch


def define_optimizer(conf, model, optimizer_name, lr=None):
    #define the param to optimize.
    no_decay = []
    #no_decay = ['conv1_u', 'conv1_v', 'conv2_u', 'conv2_v','conv3_u','conv3_v']
    params = [
        {
            "params": [value],
            "name": key,
            "weight_decay": conf.weight_decay if not any(nd in key for nd in no_decay) else 0.0,
            "param_size": value.size(),
            "nelement": value.nelement(),
        }
        for key, value in model.named_parameters()
    ]
    # define the optimizer.
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr = conf.lr if lr is None else lr,
            momentum=conf.momentum_factor,
            nesterov=conf.use_nesterov,
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=conf.lr if lr is None else lr
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=conf.lr if lr is None else lr
        )
    else:
        raise NotImplementedError
    return optimizer
