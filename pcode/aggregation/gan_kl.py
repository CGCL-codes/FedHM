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
        val_data_loader,
        test_data_loader,
):
    fl_aggregate = conf.fl_aggregate

    # selected_gan_models = [gan_models[s] for s in selected_client_ids]
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
        kt = GanKTSolver(
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
            batch_size=128,
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
            student_learning_rate=2e-3
            if "student_learning_rate" not in fl_aggregate
            else fl_aggregate["student_learning_rate"],
            AT_beta=0 if "AT_beta" not in fl_aggregate else fl_aggregate["AT_beta"],
            KL_temperature=1
            if "temperature" not in fl_aggregate
            else fl_aggregate["temperature"],
            log_fn=conf.logger.log,
            eval_batches_freq=6
            if "eval_batches_freq" not in fl_aggregate
            else int(fl_aggregate["eval_batches_freq"]),
            early_stopping_server_batches=60
            if "early_stopping_server_batches" not in fl_aggregate
            else int(fl_aggregate["early_stopping_server_batches"]),
            update_student_scheme="avg_losses"
            if "update_student_scheme" not in fl_aggregate
            else fl_aggregate["update_student_scheme"],
            server_teaching_scheme=None
            if "server_teaching_scheme" not in fl_aggregate
            else fl_aggregate["server_teaching_scheme"],
            return_best_model_on_val=True
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
import math

class Generator(nn.Module):
    def __init__(self, nz=128, ngf=64, nc=3, img_size=32):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False)
        )

    def forward(self, z):
        out = self.l1(z.view(z.shape[0], -1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        try:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
        except:
            pass

class GanKTSolver(object):
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
            eval_batches_freq=60,
            early_stopping_server_batches=1000,
            update_student_scheme="avg_losses",  # either avg_losses or avg_logits
            server_teaching_scheme=None,
            return_best_model_on_val=False,
            beta=0.1,
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

        self.generator = self.prepare_gen(Generator(nz=100, nc=3, img_size=32))
        student_model = student_model.cuda()
        self.init_server_student = copy.deepcopy(student_model)

        self.server_student = self.prepare_model(
            student_model, self.device, is_teacher=False
        )
        self.client_teachers = [
            self.prepare_model(
                _teacher, self.device, is_teacher=True
            )
            for _teacher in teacher_models
        ]
        self.return_best_model_on_val = return_best_model_on_val


        # init the loaders.
        self.val_data_loader = val_data_loader

        # init the optimizers.
        self.optimizer_server_student = optim.SGD(
            self.server_student.parameters(), lr = 0.1, weight_decay=5e-4, momentum = 0.9
        )
        self.optimizer_generator = optim.Adam(
            self.generator.parameters(), lr=1e-3
        )
        # init the training scheduler.
        self.server_teaching_scheme = server_teaching_scheme
        self.use_server_model_scheduler = use_server_model_scheduler
        self.scheduler_server_student = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_server_student,
            self.total_n_server_pseudo_batches ,
            last_epoch=-1,
        )
        self.scheduler_generator = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_generator,
            self.total_n_server_pseudo_batches ,
            last_epoch=-1
        )

        # For distillation.
        self.AT_beta = AT_beta
        self.beta = beta
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
        server_best_tracker = BestPerf(best_perf=None, larger_is_better=True)
        # iterate over dataset
        batch_count = 0
        best_models = [None]

        init_perf_on_val = self.validate(
            model=self.init_server_student, data_loader=self.val_data_loader
        )
        self.log_fn(
            f"Batch {batch_count}/{self.total_n_server_pseudo_batches}: Student Validation Acc={init_perf_on_val}."
        )
        # warm up
        # for b in range(10):
        #     z = torch.randn((self.batch_size, 100, 1, 1)).cuda()
        #     pseudo_data = self.generator(z)
        #     teacher_logits = 0
        #     feature_loss = 0
        #     for i, _teacher in enumerate(self.client_teachers):
        #         feature, logit = _teacher(pseudo_data)
        #         teacher_logits = logit + teacher_logits
        #         feature_loss = feature_loss - feature.abs().mean()
        #
        #     student_feature, student_logits = self.server_student(pseudo_data)
        #     feature_loss = feature_loss / self.numb_teachers
        #     loss_g = -F.l1_loss(student_logits, teacher_logits * (1.0 / self.numb_teachers)) + self.beta * feature_loss
        #     self.optimizer_generator.zero_grad()
        #     loss_g.backward()
        #     self.optimizer_generator.step()
        # self.scheduler_generator.step()

        while batch_count < self.total_n_server_pseudo_batches:

            z = torch.randn((self.batch_size, 100, 1, 1)).cuda()
            pseudo_data = self.generator(z)
            teacher_logits = 0
            feature_loss = 0
            for i, _teacher in enumerate(self.client_teachers):
                feature, logit = _teacher(pseudo_data)
                teacher_logits = logit + teacher_logits
                feature_loss = feature_loss - feature.abs().mean()

            student_feature, student_logits = self.server_student(pseudo_data)
            feature_loss = feature_loss / self.numb_teachers
            teacher_logits = teacher_logits / self.numb_teachers
            loss_g = -F.l1_loss(student_logits, teacher_logits) + self.beta * feature_loss
            if loss_g <= -20:
                self.log_fn(f"gan diverge, current loss_g:{loss_g:.4f}")
                break
            # loss_g = loss_g / self.numb_teachers
            self.optimizer_generator.zero_grad()
            loss_g.backward()
            # torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5)
            self.optimizer_generator.step()

            for student_iter in range(5):
                # steps on the same pseudo data
                z = torch.randn((self.batch_size, 100, 1, 1)).cuda()
                pseudo_data = self.generator(z).detach()

                # get the logits.
                with torch.no_grad():
                    teacher_logits = 0
                    for i, _teacher in enumerate(self.client_teachers):
                        feature, logit = _teacher(pseudo_data)
                        teacher_logits = logit + teacher_logits

                    teacher_logits = teacher_logits * (1.0 / self.numb_teachers)
                student_feature, student_logits = self.server_student(pseudo_data)
                loss_s = F.l1_loss(student_logits, teacher_logits)

                # loss_s = loss_s / self.numb_teachers
                self.optimizer_server_student.zero_grad()
                loss_s.backward()
                # torch.nn.utils.clip_grad_norm_(self.server_student.parameters(), 5)
                self.optimizer_server_student.step()

                # after each batch.
            if self.use_server_model_scheduler:
                self.scheduler_server_student.step()
                self.scheduler_generator.step()

            batch_count += 6

            if batch_count % 60 == 0:
                self.log_fn(
                    f" loss_g: {loss_g:.4f}, loss_s: {loss_s:.4f}"
                )
                validated_perf = self.validate(
                    model=self.server_student, data_loader=self.val_data_loader
                )
                self.log_fn(
                    f"Batch {batch_count}/{self.total_n_server_pseudo_batches}: Student Loss={server_tracker.stat['student_loss'].avg:02.5f}; Student Validation Acc={validated_perf}."
                )
                server_tracker.reset()

                # check early stopping.
                if self.base_solver.check_early_stopping(
                        model=self.server_student,
                        model_ind=0,
                        best_tracker=server_best_tracker,
                        validated_perf=validated_perf,
                        validated_perfs=self.validated_perfs,
                        perf_index=batch_count + 1,
                        early_stopping_batches=self.early_stopping_server_batches,
                        best_models=best_models,
                ):
                    break
            # update the tracker after each batch.
            server_tracker.update_metrics([loss_s], n_samples=self.batch_size)
        use_init_server_model = True
        if self.return_best_model_on_val and server_best_tracker.best_perf is not None:
            use_init_server_model = (
                True
                if init_perf_on_val["top1"] > server_best_tracker.best_perf
                else False
            )

        # get the server model.
        if use_init_server_model:
            self.log_fn("use init server model instead.")
            best_server_dict = self.init_server_student.state_dict()
        else:
            best_server_dict = best_models[0]

        # update the server model.
        self.server_student.load_state_dict(best_server_dict)
        self.server_student = self.server_student.cpu()

    def prepare_model(self, model, device, is_teacher):
        model = model.to(device)
        model = copy.deepcopy(model)
        if is_teacher:
            model.requires_grad_(False)
            model.eval()
        else:
            #model.apply(weights_init_normal)
            model.requires_grad_(True)
            model.train()
        return model

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

    def prepare_gen(self, model):
        model = model.cuda()
        model.apply(weights_init_normal)
        model.requires_grad_(True)
        model.train()
        return model
