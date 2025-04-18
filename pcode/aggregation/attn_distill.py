# -*- coding: utf-8 -*-
import copy
import collections

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from pcode.aggregation.adv_knowledge_transfer import BaseKTSolver
import pcode.aggregation.utils as agg_utils
from pcode.utils.stat_tracker import RuntimeTracker, BestPerf
import pcode.master_utils as master_utils


def get_unlabeled_data(fl_aggregate, distillation_data_loader):
    if (
            "use_data_scheme" not in fl_aggregate
            or fl_aggregate["use_data_scheme"] == "real_data"
    ):
        return distillation_data_loader
    else:
        return None


def aggregate(
        conf,
        fedavg_models,
        client_models,
        criterion,
        metrics,
        flatten_local_models,
        fa_val_perf,
        distillation_sampler,
        distillation_data_loader,
        val_data_loader,
        test_data_loader,
):
    fl_aggregate = conf.fl_aggregate

    # recover the models on the computation device.
    _, local_models = agg_utils.recover_models(
        conf, client_models, flatten_local_models
    )

    # include model from previous comm. round.
    if (
            "include_previous_models" in fl_aggregate
            and fl_aggregate["include_previous_models"] > 0
    ):
        local_models = agg_utils.include_previous_models(conf, local_models)

    # evaluate the local model on the test_loader
    if "eval_local" in fl_aggregate and fl_aggregate["eval_local"]:
        perfs = []
        for idx, local_model in enumerate(local_models.values()):
            conf.logger.log(f"Evaluate the local model-{idx}.")
            perf = master_utils.validate(
                conf,
                coordinator=None,
                model=local_model,
                criterion=criterion,
                metrics=metrics,
                data_loader=test_data_loader,
                label=None,
                display=False,
            )
            perfs.append(perf["top1"])
        conf.logger.log(
            f"The averaged test performance of the local models: {sum(perfs) / len(perfs)}; the details of the local performance: {perfs}."
        )

    # evaluate the ensemble of local models on the test_loader
    if "eval_ensemble" in fl_aggregate and fl_aggregate["eval_ensemble"]:
        master_utils.ensembled_validate(
            conf,
            coordinator=None,
            models=list(local_models.values()),
            criterion=criterion,
            metrics=metrics,
            data_loader=test_data_loader,
            label="ensemble_test_loader",
            ensemble_scheme=None
            if "update_student_scheme" not in fl_aggregate
            else fl_aggregate["update_student_scheme"],
        )

    # distillation.
    _client_models = {}
    for arch, fedavg_model in fedavg_models.items():
        conf.logger.log(
            f"Master: we have {len(local_models)} local models for noise distillation (use {arch} for the distillation)."
        )
        kt = AttnKTSolver(
            conf=conf,
            teacher_models=list(local_models.values()),
            student_model=fedavg_model
            if "use_fedavg_as_start" not in fl_aggregate
            else (
                fedavg_model
                if fl_aggregate["use_fedavg_as_start"]
                else copy.deepcopy(client_models[arch])
            ),
            criterion=criterion,
            metrics=metrics,
            batch_size=128
            if "batch_size" not in fl_aggregate
            else int(fl_aggregate["batch_size"]),
            total_n_server_pseudo_batches=1000 * 10
            if "total_n_server_pseudo_batches" not in fl_aggregate
            else int(fl_aggregate["total_n_server_pseudo_batches"]),
            server_local_steps=1
            if "server_local_steps" not in fl_aggregate
            else int(fl_aggregate["server_local_steps"]),
            val_data_loader=val_data_loader,
            use_server_model_scheduler=True
            if "use_server_model_scheduler" not in fl_aggregate
            else fl_aggregate["use_server_model_scheduler"],
            same_noise=True
            if "same_noise" not in fl_aggregate
            else fl_aggregate["same_noise"],
            student_learning_rate=1e-3
            if "student_learning_rate" not in fl_aggregate
            else fl_aggregate["student_learning_rate"],
            AT_beta=0 if "AT_beta" not in fl_aggregate else fl_aggregate["AT_beta"],
            KL_temperature=1
            if "temperature" not in fl_aggregate
            else fl_aggregate["temperature"],
            log_fn=conf.logger.log,
            eval_batches_freq=100
            if "eval_batches_freq" not in fl_aggregate
            else int(fl_aggregate["eval_batches_freq"]),
            early_stopping_server_batches=2000
            if "early_stopping_server_batches" not in fl_aggregate
            else int(fl_aggregate["early_stopping_server_batches"]),
            update_student_scheme="avg_losses"
            if "update_student_scheme" not in fl_aggregate
            else fl_aggregate["update_student_scheme"],
            server_teaching_scheme=None
            if "server_teaching_scheme" not in fl_aggregate
            else fl_aggregate["server_teaching_scheme"],
            return_best_model_on_val=False
            if "return_best_model_on_val" not in fl_aggregate
            else fl_aggregate["return_best_model_on_val"],
        )
        kt.distillation()
        _client_models[arch] = kt.server_student.cpu()

    # update local models from the current comm. round.
    if (
            "include_previous_models" in fl_aggregate
            and fl_aggregate["include_previous_models"] > 0
    ):
        agg_utils.update_previous_models(conf, _client_models)

    # free the memory.
    del local_models, kt
    torch.cuda.empty_cache()
    return _client_models


import torch.nn as nn


class Aggregation(nn.Module):
    def __init__(self,channel_dim,factor = 8):
        super(Aggregation, self).__init__()
        self.query_linear = nn.Linear(channel_dim, channel_dim // factor)
        self.key_linear = nn.Linear(channel_dim, channel_dim // factor)

    def forward(self, teacher_logits, teacher_features):
        # [B,n,C]
        query = self.query_linear(teacher_features)
        key = self.key_linear(teacher_features)
        energy = torch.bmm(query, key.permute(0, 2, 1))
        attention = nn.functional.softmax(energy, dim=-1)
        # [B,Classes,n] X [B,n,n] = [B,classes,n]
        x_m = torch.bmm(teacher_logits, attention.permute(0, 2, 1))

        return x_m

class Attention(nn.Module):
    def __init__(self,channel_dim,factor = 8):
        super(Attention, self).__init__()
        #self.query_linear = nn.Linear(channel_dim, channel_dim // factor)
        #self.key_linear = nn.Linear(channel_dim, channel_dim // factor)

    def forward(self, student_features,teacher_logits, teacher_features):
        # [B,n,C]
        query = student_features.unsqueeze(1)
        key = teacher_features
        #query = self.query_linear(student_features).unsqueeze(1)
        #key = self.key_linear(teacher_features)
        energy = -torch.bmm(query, key.permute(0, 2, 1))
        attention = nn.functional.softmax(energy, dim=-1)
        # [B,Classes,n] X [B,n,n] = [B,classes,n]
        x_m = torch.bmm(teacher_logits, attention.permute(0, 2, 1))
        x_m = torch.squeeze(x_m)
        return x_m

class AttnKTSolver(object):
    """ Main solver class to transfer the knowledge through noise or unlabelled data."""

    def __init__(
            self,
            conf,
            teacher_models,
            student_model,
            criterion,
            metrics,
            batch_size,
            total_n_server_pseudo_batches=0,
            server_local_steps=1,
            val_data_loader=None,
            use_server_model_scheduler=True,
            same_noise=True,
            student_learning_rate=1e-3,
            AT_beta=0,
            KL_temperature=1,
            log_fn=print,
            eval_batches_freq=100,
            early_stopping_server_batches=1000,
            update_student_scheme="avg_losses",  # either avg_losses or avg_logits
            server_teaching_scheme=None,
            return_best_model_on_val=False,
    ):
        # general init.
        self.conf = conf
        self.device = (
            torch.device("cuda") if conf.graph.on_cuda else torch.device("cpu")
        )

        # init the validation criterion and metrics.
        self.criterion = criterion
        self.metrics = metrics

        # init training logics and loaders.
        self.same_noise = same_noise
        self.batch_size = batch_size
        self.total_n_server_pseudo_batches = total_n_server_pseudo_batches
        self.server_local_steps = server_local_steps

        # init the fundamental solver.
        self.base_solver = BaseKTSolver(KL_temperature=KL_temperature, AT_beta=AT_beta)

        # init student and teacher nets.
        self.numb_teachers = len(teacher_models)

        self.aggregation = Aggregation(64)
        self.aggregation = self.aggregation.cuda()
        self.aggregation.requires_grad_(True)

        self.server_student = self.base_solver.prepare_model(
            conf, student_model, self.device, _is_teacher=False
        )
        self.client_teachers = [
            self.base_solver.prepare_model(
                conf, _teacher, self.device, _is_teacher=True
            )
            for _teacher in teacher_models
        ]
        self.return_best_model_on_val = return_best_model_on_val
        self.init_server_student = copy.deepcopy(self.server_student)

        # init the loaders.
        #self.val_data_loader = val_data_loader
        #self.distillation_sampler = distillation_sampler
        # self.distillation_data_loader = self.preprocess_unlabeled_real_data(
        #     distillation_sampler, distillation_data_loader
        # )
        self.distillation_data_loader = val_data_loader
        # teacher_params = list([])
        # for teacher in self.client_teachers:
        #     teacher_params = teacher_params + list(teacher.parameters())

        # init the optimizers.
        self.optimizer_server_student = optim.AdamW(
            list(self.server_student.parameters()), lr=student_learning_rate
            #+teacher_params+list(self.aggregation.parameters())
        )

        # init the training scheduler.
        self.server_teaching_scheme = server_teaching_scheme
        self.use_server_model_scheduler = use_server_model_scheduler
        self.scheduler_server_student = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_server_student,
            self.total_n_server_pseudo_batches,
            last_epoch=-1,
        )

        # For distillation.
        self.AT_beta = AT_beta
        self.KL_temperature = KL_temperature

        # Set up & Resume
        self.log_fn = log_fn
        self.eval_batches_freq = eval_batches_freq
        self.early_stopping_server_batches = early_stopping_server_batches
        self.update_student_scheme = update_student_scheme
        self.validated_perfs = collections.defaultdict(list)
        print("\tFinished the initialization for NoiseKTSolver.")

    """related to distillation."""

    def distillation(self):
        # init the tracker.
        server_tracker = RuntimeTracker(
            metrics_to_track=["student_loss"], force_to_replace_metrics=True
        )
        import copy
        teacher_copy = copy.deepcopy(self.client_teachers)
        # iterate over dataset
        for epoch in range(self.numb_teachers):
            for pseudo_data,target in self.distillation_data_loader:
                pseudo_data,target = pseudo_data.cuda(),target.cuda()
                # steps on the same pseudo data

                student_feature, student_logits = self.server_student(pseudo_data)
                # get the logits.
                with torch.no_grad():
                    teacher_features, teacher_logits = [], []
                    for _teacher in self.client_teachers:
                        feature, logit = _teacher(pseudo_data)
                        teacher_features.append(feature)
                        teacher_logits.append(logit)

                    teacher_logits = torch.stack(teacher_logits, dim=-1)
                    teacher_features = torch.stack(teacher_features, dim=1)
                    weighted_logits = self.aggregation(student_feature,teacher_logits, teacher_features)

                # loss_group = 0
                # loss_true = 0
                # for i in range(self.numb_teachers):
                #     loss_true += nn.functional.cross_entropy(teacher_logits[:,:,i],target)
                #     loss_group += self.base_solver.divergence(teacher_logits[:, :, i], weighted_logits[:, :, i])

                loss = F.cross_entropy(student_logits,target) + self.base_solver.divergence(student_logits,weighted_logits)
                # loss = F.cross_entropy(student_logits,target) + self.base_solver.divergence(student_logits, torch.mean(teacher_logits, dim=-1)) \
                #        + loss_true + loss_group

                self.optimizer_server_student.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.server_student.parameters(), 5)
                self.optimizer_server_student.step()


                # after each batch.
                if self.use_server_model_scheduler:
                    self.scheduler_server_student.step()

            # update the tracker after each batch.
            server_tracker.update_metrics([loss], n_samples=self.batch_size)

        self.server_student = self.server_student.cpu()

    def validate(
            self, model, data_loader=None, criterion=None, metrics=None, device=None
    ):
        if data_loader is None:
            return -1
        else:
            val_perf = master_utils.validate(
                conf=self.conf,
                coordinator=None,
                model=model,
                criterion=self.criterion if criterion is None else criterion,
                metrics=self.metrics if metrics is None else metrics,
                data_loader=data_loader,
                label=None,
                display=False,
            )
            model = model.to(self.device if device is None else device)
            model.train()
            return val_perf

    """related to dataset used for distillation."""

    def preprocess_unlabeled_real_data(
            self, distillation_sampler, distillation_data_loader
    ):
        if (
                "noise_kt_preprocess" not in self.conf.fl_aggregate
                or not self.conf.fl_aggregate["noise_kt_preprocess"]
        ):
            return distillation_data_loader

        # preprocessing for noise_kt.
        self.log_fn(f"preprocessing the unlabeled data.")

        ## prepare the data_loader.
        data_loader = torch.utils.data.DataLoader(
            distillation_sampler.use_indices(),
            batch_size=self.conf.batch_size,
            shuffle=False,
            num_workers=self.conf.num_workers,
            pin_memory=self.conf.pin_memory,
            drop_last=False,
        )

        ## evaluate the entropy of each data point.
        outputs = []
        for _input, _ in data_loader:
            _outputs = [
                F.softmax(client_teacher(_input.to(self.device)), dim=1)
                for client_teacher in self.client_teachers
            ]
            _entropy = Categorical(sum(_outputs) / len(_outputs)).entropy()
            outputs.append(_entropy)
        entropy = torch.cat(outputs)

        ## we pick samples that have low entropy.
        assert "noise_kt_preprocess_size" in self.conf.fl_aggregate
        noise_kt_preprocess_size = int(
            self.conf.fl_aggregate["noise_kt_preprocess_size"]
        )
        _, indices = torch.topk(
            entropy,
            k=min(len(entropy), noise_kt_preprocess_size),
            largest=False,
            sorted=False,
        )
        distillation_sampler.sampled_indices = distillation_sampler.sampled_indices[
            indices.cpu()
        ]

        ## create the dataloader.
        return torch.utils.data.DataLoader(
            distillation_sampler.use_indices(),
            batch_size=self.conf.batch_size,
            shuffle=self.conf.fl_aggregate["randomness"]
            if "randomness" in self.conf.fl_aggregate
            else True,
            num_workers=self.conf.num_workers,
            pin_memory=self.conf.pin_memory,
            drop_last=False,
        )

    def _create_data_randomly(self):
        # create pseudo_data and map to [0, 1].
        pseudo_data = torch.randn(
            (self.batch_size, 3, self.conf.img_resolution, self.conf.img_resolution),
            requires_grad=False,
        ).to(device=self.device)
        pseudo_data = (pseudo_data - torch.min(pseudo_data)) / (
                torch.max(pseudo_data) - torch.min(pseudo_data)
        )

        # map values to [-1, 1] if necessary.
        if self.conf.pn_normalize:
            pseudo_data = (pseudo_data - 0.5) * 2
        return pseudo_data
