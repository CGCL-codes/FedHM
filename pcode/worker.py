# -*- coding: utf-8 -*-
import copy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.create_optimizer as create_optimizer
import pcode.create_scheduler as create_scheduler
import pcode.datasets.mixup_data as mixup
import pcode.local_training.compressor as compressor
from pcode.utils.logging import display_training_stat
from pcode.utils.stat_tracker import RuntimeTracker
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.timer import Timer


class Worker(object):
    def __init__(self, conf):
        self.conf = conf

        # some initializations.
        self.rank = conf.graph.rank
        conf.graph.worker_id = conf.graph.rank
        self.device = torch.device("cuda" if self.conf.graph.on_cuda else "cpu")

        # define the timer for different operations.
        # if we choose the `train_fast` mode, then we will not track the time.
        self.timer = Timer(
            verbosity_level=1 if conf.track_time else 0,
            log_fn=conf.logger.log_metric,
        )
        self.arch = None
        # create dataset (as well as the potential data_partitioner) for training.
        dist.barrier()
        self.dataset = create_dataset.define_dataset(conf, data=conf.data,agg_data_ratio=conf.agg_data_ratio)
        _, self.data_partitioner = create_dataset.define_data_loader(
            self.conf,
            dataset=self.dataset["train"],
            localdata_id=0,  # random id here.
            is_train=True,
            data_partitioner=None,
        )

        conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} initialized the local training data with Master."
        )

        # define the criterion.
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        # define the scheduler(global)
        _, self.global_model = create_model.define_model(
            self.conf, to_consistent_model=False,show_stat=False
        )
        self.global_optimizer = create_optimizer.define_optimizer(
            self.conf, model=self.global_model, optimizer_name=self.conf.optimizer
        )
        self.global_scheduler = create_scheduler.Scheduler(self.conf, optimizer=self.global_optimizer)

        # define the model compression operators.
        if conf.local_model_compression is not None:
            if conf.local_model_compression == "quantization":
                self.model_compression_fn = compressor.ModelQuantization(conf)

        if self.conf.split_mix:
            if self.conf.pruning:
                slimmable_ratios = [eval(arch.split('_')[-1]) for arch in self.conf.arch_info["worker"]]
            else:
                slimmable_ratios = [1.0 / float(eval(arch.split('_')[-1])) for arch in self.conf.arch_info["worker"]]

            self.max_ratio = max(slimmable_ratios)
            self.atom_slim_ratio = min(min(slimmable_ratios), self.conf.atom_slim_ratio)

        self.arch = None
        conf.logger.log(
            f"Worker-{conf.graph.worker_id} initialized dataset/criterion.\n"
        )


    def run(self):
        while True:
            self._listen_to_master()

            if self.conf.graph.client_id <= 0:
                dist.barrier()
                dist.barrier()
                dist.barrier()
                continue

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return


            self._recv_model_from_master()
            self._train()

            self._send_model_to_master(self.model)

            # check if we need to terminate the training or not.
            if self._terminate_by_complete_training():
                return

    def _listen_to_master(self):
        # listen to master, related to the function `_activate_selected_clients` in `master.py`.
        msg_len = 3
        if self.conf.split_mix:
            msg_len += 1
        if self.conf.dynamic:
            msg_len += 1

        msg = torch.zeros((msg_len, self.conf.n_participated))
        dist.broadcast(tensor=msg, src=0)

        self.conf.graph.client_id, self.conf.graph.comm_round, self.n_local_epochs = (
            msg[:3, self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()
        )

        if self.conf.split_mix:
            self.slim_length = msg[3, self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()

        if self.conf.dynamic:
            self.arch_index = msg[msg_len - 1][self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()
            archs = self.conf.arch_info["worker"]
            self.arch = archs[self.arch_index]
        else:
            self.arch = None

        if self.conf.graph.client_id > 0:
            self.get_label_split()
            # once we receive the signal, we init for the local training.
            self.arch, self.model = create_model.define_model(
                self.conf, to_consistent_model=False, client_id=self.conf.graph.client_id,arch=self.arch
            )
            self.model_state_dict = self.model.state_dict()
            self.model_tb = TensorBuffer(list(self.model_state_dict.values()))
            self.metrics = create_metrics.Metrics(self.model, task="classification")
        dist.barrier()


    def _recv_model_from_master(self):
        # related to the function `_send_model_to_selected_clients` in `master.py`
        #old_buffer = copy.deepcopy(self.model_tb.buffer)
        dist.recv(self.model_tb.buffer, src=0)
        if self.conf.split_mix:
            self.slim_infos = torch.zeros((2, self.slim_length))
            dist.recv(self.slim_infos, src = 0)
            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received slim_idx{self.slim_infos[1].to(int).cpu().numpy().tolist()}from Master."
            )

        #new_buffer = copy.deepcopy(self.model_tb.buffer)
        self.model_tb.unpack(self.model_state_dict.values())
        self.model.load_state_dict(self.model_state_dict,strict=True)

        #random_reinit.random_reinit_model(self.conf, self.model)

        if self.conf.self_distillation > 0 or self.conf.local_prox_term > 0:
            self.init_model = self._turn_off_grad(copy.deepcopy(self.model).to(self.device))

        # self.aggregation = Aggregation(self.model.classifier.in_features).cuda()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received the model ({self.arch}) from Master."
        )
        # The model status {'is updated' if old_buffer.norm() != new_buffer.norm() else 'is not updated'}
        dist.barrier()


    def _train(self):
        self._turn_on_grad()
        self.model.train()

        # init the model and dataloader.
        if self.conf.graph.on_cuda:
            self.model = self.model.to(self.device)
        self.train_loader, _ = create_dataset.define_data_loader(
            self.conf,
            dataset=self.dataset["train"],
            # localdata_id start from 0 to the # of clients - 1.
            # client_id starts from 1 to the # of clients.
            localdata_id=self.conf.graph.client_id - 1,
            is_train=True,
            data_partitioner=self.data_partitioner,
        )
        lr = self.global_optimizer.param_groups[0]['lr']

        self.optimizer = create_optimizer.define_optimizer(
            self.conf, model=self.model, optimizer_name=self.conf.optimizer,lr = lr
        )
        self.scheduler = create_scheduler.Scheduler(
            self.conf, optimizer=self.optimizer
        )

        self.tracker = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) start training (current communication rounds={self.conf.graph.comm_round}), lr = {lr}."
        )

        # efficient local training.
        if hasattr(self, "model_compression_fn"):
            self.model_compression_fn.compress_model(
                param_groups=self.optimizer.param_groups
            )
        if self.conf.local_prox_term != 0:
            self.global_weight_collector = list(self.init_model.parameters())

        if self.conf.split_mix:
            self.slim_ratios, self.slim_shifts = self.slim_infos
            self.slim_shifts = self.slim_infos[1].int()
            self.slim_ratios = [self.atom_slim_ratio] * len(self.slim_shifts)

        # entering local updates and will finish only after reaching the expected local_n_epochs.
        while True:

            for _input, _target in self.train_loader:
                # load data
                with self.timer("load_data", epoch=self.scheduler.epoch_):
                    data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target,is_training=True,device=self.device
                    )

                self.optimizer.zero_grad()
                if self.conf.split_mix:
                    with self.timer("forward_pass", epoch=self.scheduler.epoch_):
                        total_loss = 0
                        for in_slim_shift in self.slim_shifts:
                            self.model.switch_slim_mode(self.atom_slim_ratio, slim_bias_idx=in_slim_shift)
                            loss, performance, output = self._inference(data_batch)
                            loss.backward()
                            total_loss += loss

                        total_loss /= self.slim_length
                        loss = total_loss

                else:
                    # inference and get current performance.
                    with self.timer("forward_pass", epoch=self.scheduler.epoch_):
                        loss, performance, output = self._inference(data_batch)

                    if self.conf.local_prox_term != 0:
                        loss = self._local_training_with_prox(loss,output.size(0))
                    else:
                        loss = self._local_training_with_self_distillation(
                            loss, output, data_batch)
                    loss.backward()
                    self._add_frob_grad()

                with self.timer("backward_pass", epoch=self.scheduler.epoch_):

                    self.optimizer.step()
                    self.scheduler.step()

                # update tracker.
                if self.tracker is not None:
                    bsz = data_batch["target"].size(0)
                    self.tracker.update_local_metrics(
                        loss.item(), 0, n_samples=bsz
                    )
                    for idx in range(1, 1 + len(performance)):
                        self.tracker.update_local_metrics(
                            performance[idx - 1], idx, n_samples=bsz
                        )
                # efficient local training.
                with self.timer("compress_model", epoch=self.scheduler.epoch_):
                    if hasattr(self, "model_compression_fn"):
                        self.model_compression_fn.compress_model(
                            param_groups=self.optimizer.param_groups
                        )

                # display the logging info.
                #display_training_stat(self.conf, self.scheduler, self.tracker)

                # display tracking time.
                if (
                        self.conf.display_tracked_time
                        and self.scheduler.local_index % self.conf.summary_freq == 0
                ):
                    self.conf.logger.log(self.timer.summary())

                # check divergence.
                # if self.tracker.stat["loss"].avg > 1e3 or np.isnan(
                #         self.tracker.stat["loss"].vg
                # ):
                if loss > 1e3 or torch.isnan(loss):
                    self.conf.logger.log(
                        f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) diverges!!!!!Early stop it."
                    )
                    self._terminate_comm_round()
                    return

                # check stopping condition.
                if self._is_finished_one_comm_round():
                    self._terminate_comm_round()
                    return


            # display the logging info.
            display_training_stat(self.conf, self.scheduler, self.tracker)
            # refresh the logging cache at the end of each epoch.
            self.tracker.reset()
            if self.conf.logger.meet_cache_limit():
                self.conf.logger.save_json()

    def _multiple_inference(self, data_batch):
        total_loss = 0
        for slim_ratio, in_slim_shift \
                in sorted(zip(self.slim_ratios, self.slim_shifts), reverse=False, key=lambda ss_pair: ss_pair[0]):

            self.model.switch_slim_mode(slim_ratio, slim_bias_idx=in_slim_shift)
            loss,performance, output = self._inference(data_batch)
            loss.backward()
            total_loss += loss

        return total_loss / self.slim_length


    def _inference(self, data_batch):
        """Inference on the given model and get loss and accuracy."""
        # do the forward pass and get the output.
        output = self.model(data_batch["input"])

        if self.conf.mask:
            label_mask = torch.zeros(self.conf.num_classes, device=self.device)
            label_mask[self.label_split] = 1
            output = output.masked_fill(label_mask == 0, 0)

        # evaluate the output and get the loss, performance.
        if self.conf.use_mixup:
            loss = mixup.mixup_criterion(
                self.criterion,
                output,
                data_batch["target_a"],
                data_batch["target_b"],
                data_batch["mixup_lambda"],
            )

            performance_a = self.metrics.evaluate(loss, output, data_batch["target_a"])
            performance_b = self.metrics.evaluate(loss, output, data_batch["target_b"])
            performance = [
                data_batch["mixup_lambda"] * _a + (1 - data_batch["mixup_lambda"]) * _b
                for _a, _b in zip(performance_a, performance_b)
            ]
        else:
            loss = self.criterion(output, data_batch["target"])
            performance = self.metrics.evaluate(loss, output, data_batch["target"])


        return loss, performance, output

    def _add_grad_from_prox_regularized_loss(self):
        assert self.conf.local_prox_term >= 0
        if self.conf.local_prox_term != 0:
            assert self.conf.weight_decay == 0
            assert self.conf.optimizer == "sgd"
            assert self.conf.momentum_factor == 0

            for _param, _init_param in zip(
                    self.model.parameters(), self.init_model.parameters()
            ):
                if _param.grad is not None:
                    _param.grad.data.add_(
                        (_param.data - _init_param.data) * self.conf.local_prox_term
                    )

    def apply_frobdecay(self,name, module, skiplist=[]):
        if hasattr(module, 'frobgrad'):
            return not any(name[-len(entry):] == entry for entry in skiplist)
        return False

    def _add_frob_grad(self):
        if self.conf.weight_decay > 0:
            for i, module in enumerate(module for name, module in self.model.named_modules()
                                       if self.apply_frobdecay(name, module, skiplist=[])):
                module.frobgrad(coef=self.conf.weight_decay)


    def _local_training_with_prox(self, loss, bsz):
        assert self.global_weight_collector is not None

        loss2 = 0
        for param_index, param in enumerate(self.model.parameters()):
            loss2 += ((self.conf.local_prox_term / 2) * torch.norm(
                (param - self.global_weight_collector[param_index])) ** 2)

        if self.tracker is not None:
            self.tracker.update_local_metrics(
                loss2.item(), -1, n_samples=bsz
            )
        loss = loss + loss2

        return loss


    def _local_training_with_self_distillation(self, loss, output, data_batch):
        if self.conf.self_distillation > 0 and self.conf.graph.comm_round > 1:
            with torch.no_grad():
                _, teacher_logits = self.init_model(data_batch["input"])
            if self.conf.loss_type == 'kl':
                loss2 = self.conf.self_distillation * self._divergence(
                    student_logits=output / self.conf.self_distillation_temperature,
                    teacher_logits=teacher_logits / self.conf.self_distillation_temperature,
                )
            else:
                loss2 = self.conf.self_distillation * F.mse_loss(output / self.conf.self_distillation_temperature,
                                                             teacher_logits / self.conf.self_distillation_temperature)
            if self.conf.AT_beta > 0:
                at_loss = 0
                student_activations = self.model.activations
                teacher_activations = self.init_model.activations
                for i in range(len(student_activations)):
                    at_loss = at_loss + self.conf.AT_beta * self.attention_diff(
                        student_activations[i], teacher_activations[i]
                    )
                loss2 += at_loss
            loss = loss + loss2

            if self.tracker is not None:
                self.tracker.update_local_metrics(
                    loss2.item(), -1, n_samples=data_batch["target"].size(0)
                )
        return loss

    def _divergence(self, student_logits, teacher_logits):
        divergence = self.conf.self_distillation_temperature * self.conf.self_distillation_temperature * F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction="batchmean",
        )  # forward KL
        return divergence

    def attention(self, x):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        """
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def attention_diff(self, x, y):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        :param y = activations
        """
        return (self.attention(x) - self.attention(y)).pow(2).mean()

    def similarity(self, x1, x2):
        sim = F.cosine_similarity(x1, x2, dim=-1)
        return sim

    def _turn_on_grad(self):
        self.model.requires_grad_(True)


    def _turn_off_grad(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _send_model_to_master(self,model):
        dist.barrier()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) sending the model ({self.arch}) back to Master."
        )
        if self.conf.split_mix:
            model.switch_slim_mode(self.max_ratio)
        flatten_model = TensorBuffer(list(model.state_dict().values()))
        dist.send(tensor=flatten_model.buffer, dst=0)
        dist.barrier()

    def _terminate_comm_round(self):

        # history_model = self._turn_off_grad(copy.deepcopy(self.model).cuda())
        self.model = self.model.cpu()

        # if self.conf.global_history:
        #     for i in range(len(self.global_models_buffer)):
        #         self.global_models_buffer[i] = self.global_models_buffer[i].cpu()
        self.scheduler.clean()
        self.global_scheduler.lr_scheduler.step()
        self.conf.logger.save_json()
        torch.cuda.empty_cache()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) finished one round of federated learning: (comm_round={self.conf.graph.comm_round})."
        )

    def _terminate_by_early_stopping(self):
        if self.conf.graph.comm_round == -1:
            dist.barrier()
            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} finished the federated learning by early-stopping."
            )
            return True
        else:
            return False

    def _terminate_by_complete_training(self):
        if self.conf.graph.comm_round == self.conf.n_comm_rounds:
            dist.barrier()
            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} finished the federated learning: (total comm_rounds={self.conf.graph.comm_round})."
            )
            return True
        else:
            return False

    def _is_finished_one_comm_round(self):
        return True if self.conf.epoch_ >= self.n_local_epochs else False

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def frobenius_norm(self,U, VT):
        m, r = U.shape
        r, n = VT.shape
        if m * n * r < r * r * (m + n):
            return torch.norm(torch.matmul(U, VT))
        return torch.sqrt(torch.trace(torch.chain_matmul(VT, VT.T, U.T, U)))

    def get_label_split(self):
        unique_elements = [stat_info[0] for stat_info in self.data_partitioner.targets_of_partitions[self.conf.graph.client_id - 1]]
        self.label_split = unique_elements
        #print(self.conf.graph.client_id, self.label_split)

