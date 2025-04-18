# -*- coding: utf-8 -*-
import copy
import os
import numpy as np
import torch
import torch.distributed as dist
import pcode.create_aggregator as create_aggregator
import pcode.create_coordinator as create_coordinator
import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.master_utils as master_utils
import pcode.utils.checkpoint as checkpoint
import pcode.utils.cross_entropy as cross_entropy
from pcode.aggregation import svd_agg, pruning_agg, mix_agg
from pcode.utils.early_stopping import EarlyStoppingTracker
from pcode.utils.tensor_buffer import TensorBuffer


class Master(object):
    def __init__(self, conf):
        self.conf = conf

        # some initializations.
        self.client_ids = list(range(1, 1 + conf.n_clients))
        self.world_ids = list(range(1, 1 + conf.n_participated))

        # create model as well as their corresponding state_dicts.
        _, self.master_model = create_model.define_model(
            conf, to_consistent_model=False, on_cuda=self.conf.on_cuda
        )

        self.used_client_archs = self.conf.arch_info["worker"]
        self.conf.used_client_archs = self.used_client_archs

        conf.logger.log(f"The client will use archs={self.used_client_archs}.")
        conf.logger.log("Master created model templates for client models.")

        self.client_models = dict(
            create_model.define_model(conf, to_consistent_model=False, arch=arch, on_cuda=self.conf.on_cuda)
            for arch in self.used_client_archs
        )
        # self.get_client_model_weights()
        self.clientid2arch = dict(
            (
                client_id,
                create_model.determine_arch(
                    conf, client_id=client_id, use_complex_arch=True
                ),
            )
            for client_id in range(1, 1 + conf.n_clients)
        )
        self.conf.clientid2arch = self.clientid2arch
        dist.barrier()
        # create dataset (as well as the potential data_partitioner) for training.
        self.dataset = create_dataset.define_dataset(conf, data=conf.data, agg_data_ratio=conf.agg_data_ratio)
        _, self.data_partitioner = create_dataset.define_data_loader(
            self.conf,
            dataset=self.dataset["train"],
            localdata_id=0,  # random id here.
            is_train=True,
            data_partitioner=None,
        )
        self.label_split = self.get_label_split()

        # create the aggregator and initialize the compressed model properly
        if len(self.used_client_archs[-1].split('_')) > 1:
            factor = eval(self.used_client_archs[-1].split('_')[-1])
            if self.conf.split_mix:
                self.hetero_agg = mix_agg.MixAggregator(conf, self.master_model, self.client_models, self.label_split)
            elif factor < 1:
                self.hetero_agg = pruning_agg.HeteroAggregator(conf, self.master_model, self.client_models, self.label_split)
                print("initilized by splitting model")
            elif factor > 1:
                self.hetero_agg = svd_agg.SVDAggregator(conf, self.master_model, self.client_models, self.label_split)
                print("initialized by spectual!")
            else:
                print("the clients failed to be sorted")
                exit(-1)

            if 'vit' not in self.conf.arch_info["master"]:
                self.client_models = self.hetero_agg.split_model(self.master_model, self.client_models)

        conf.logger.log(
            f"Master initialize the clientid2arch mapping relations: {self.clientid2arch}."
        )

        if self.conf.freeze_bn:
            self.data_loader = torch.utils.data.DataLoader(
                self.dataset["train"],
                batch_size=conf.batch_size,
                shuffle=False,
                num_workers=conf.num_workers,
                pin_memory=conf.pin_memory,
                drop_last=False
            )
            self.conf.freeze_bn = False
            modified_before = self.conf.need_scaler
            # Master.conf is pocself.test_modelsessed only in master, so it won't effect the worker's model
            self.conf.need_scaler = False
            self.test_models = dict(
                (arch, create_model.define_model(self.conf, to_consistent_model=False, arch=arch)[1].cuda())
                for arch in self.used_client_archs)
            self.conf.freeze_bn = True
            self.conf.need_scaler = modified_before


        conf.logger.log(f"Master initialized the local training data with workers.")

        # create val loader.
        # right now we just ignore the case of partitioned_by_user.
        if self.dataset["val"] is not None:
            assert not conf.partitioned_by_user
            self.val_loader, _ = create_dataset.define_data_loader(
                conf, self.dataset["val"], is_train=False
            )
            conf.logger.log(f"Master initialized val data.")
        else:
            self.val_loader = None

        # create test loaders.
        # localdata_id start from 0 to the # of clients - 1. client_id starts from 1 to the # of clients.
        if conf.partitioned_by_user:
            self.test_loaders = []
            for localdata_id in self.client_ids:
                test_loader, _ = create_dataset.define_data_loader(
                    conf,
                    self.dataset["test"],
                    localdata_id=localdata_id - 1,
                    is_train=False,
                    shuffle=False,
                )
                self.test_loaders.append(copy.deepcopy(test_loader))
        else:
            test_loader, _ = create_dataset.define_data_loader(
                conf, self.dataset["test"], is_train=False
            )
            self.test_loaders = [test_loader]

        # define the criterion and metrics.
        self.criterion = cross_entropy.CrossEntropyLoss(reduction="mean")
        self.metrics = create_metrics.Metrics(self.master_model, task="classification")
        conf.logger.log(f"Master initialized model/dataset/criterion/metrics.")

        # define the aggregators.
        self.aggregator = create_aggregator.Aggregator(
            conf,
            master_model=self.master_model,
            client_models=self.client_models,
            criterion=self.criterion,
            metrics=self.metrics,
            dataset=self.dataset,
            test_loaders=self.test_loaders,
            clientid2arch=self.clientid2arch
        )

        # if len(self.used_client_archs) > 1:
        self.coordinator = [create_coordinator.Coordinator(conf, self.metrics) for _ in
                            range(len(self.used_client_archs))]
        # else: self.coordinato
        #     self.coordinator = create_coordinator.Coordinator(conf, self.metrics)
        conf.logger.log(f"Master initialized the aggregator/coordinator.\n")

        # define early_stopping_tracker.
        self.early_stopping_tracker = EarlyStoppingTracker(
            patience=conf.early_stopping_rounds
        )

        # save arguments to disk.
        conf.is_finished = False
        checkpoint.save_arguments(conf)

    def run(self):
        to_send_history = False
        for comm_round in range(1, self.conf.n_comm_rounds + 1):
            self.conf.graph.comm_round = comm_round
            self.conf.logger.log(
                f"Master starting one round of federated learning: (comm_round={comm_round})."
            )

            # get random n_local_epochs.
            list_of_local_n_epochs = get_n_local_epoch(
                conf=self.conf, n_participated=self.conf.n_participated
            )
            self.list_of_local_n_epochs = list_of_local_n_epochs

            # random select clients from a pool.
            if self.conf.dynamic:
                self.sample_client2arch()

            selected_client_ids = self._random_select_clients()
            #selected_client_ids = self._uniform_select_clients()

            # detect early stopping.
            self._check_early_stopping()
            if self.conf.contrastive or self.conf.local_history:
                to_send_history = comm_round > 1

            self._activate_selected_clients(
                selected_client_ids, self.conf.graph.comm_round, list_of_local_n_epochs
            )

            # will decide to send the model or stop the training.
            if not self.conf.is_finished:
                # broadcast the model to activated clients.
                self._send_model_to_selected_clients(selected_client_ids, to_send_history)

            else:
                dist.barrier()
                self.conf.logger.log(
                    f"Master finished the federated learning by early-stopping: (current comm_rounds={comm_round}, total_comm_rounds={self.conf.n_comm_rounds})"
                )
                return

            # wait to receive the local models.
            flatten_local_models = self._receive_models_from_selected_clients(
                selected_client_ids
            )

            # aggregate the local models and evaluate on the validation dataset.
            self._aggregate_model_and_evaluate(flatten_local_models, selected_client_ids)

            # evaluate the aggregated model.
            self.conf.logger.log(f"Master finished one round of federated learning.\n")

        # formally stop the training (the master has finished all communication rounds).
        dist.barrier()
        self._finishing()


    def _random_select_clients(self):
        selected_client_ids = self.conf.random_state.choice(
            self.client_ids, self.conf.n_participated, replace=False
        ).tolist()
        selected_client_ids.sort()
        self.conf.logger.log(
            f"Master selected {self.conf.n_participated} from {self.conf.n_clients} clients: {selected_client_ids}."
        )
        return selected_client_ids

    def _uniform_select_clients(self):
        num_per_device = int(self.conf.arch_info["num_clients_per_model"])
        group_size = self.conf.n_clients // len(self.used_client_archs)
        select_group_size = self.conf.n_participated // len(self.used_client_archs)
        index_start = [i * num_per_device for i in range(len(self.used_client_archs))]

        selected_client_ids = []
        for i in range(len(self.used_client_archs)):
            selected_client_ids.extend(self.conf.random_state.choice(
                self.client_ids[index_start[i]: index_start[i] + group_size], select_group_size, replace=False
            ))
        selected_client_ids.sort()
        return selected_client_ids

    def _select_strong_clients(self, comm_round):
        strong_devices = self.client_ids[:int(self.conf.arch_info["num_clients_per_model"])]
        if len(strong_devices) >= self.conf.n_participated:
            upper_bound = (self.conf.n_participated * comm_round) % len(strong_devices)

            if upper_bound < self.conf.n_participated:
                upper_bound = len(strong_devices)

            selected_client_ids = self.client_ids[upper_bound - self.conf.n_participated: upper_bound]
            selected_client_ids.sort()
            self.conf.logger.log(
                f"Master selected {self.conf.n_participated} strong devices from {self.conf.n_clients} clients: {selected_client_ids}."
            )
        else:
            selected_client_ids = strong_devices
            self.conf.logger.log(
                f"Master selected {len(strong_devices)} strong devices from {self.conf.n_clients} clients: {selected_client_ids}."
            )
        # selected_client_ids = self.conf.random_state.choice(
        #     strong_devices, self.conf.n_participated, replace=False
        # ).tolist()

        return selected_client_ids

    def sample_client2arch(self):
        # rate = float(self.conf.arch_info["num_clients_per_model"]) / float(self.conf.n_clients)
        #bucket_count = int(self.conf.n_clients / self.conf.arch_info["num_clients_per_model"])
        #rate = float(self.conf.arch_info["num_clients_per_model"] / self.conf.n_clients)
        assert  isinstance(self.conf.arch_info["num_clients_per_model"], str)

        rate = [float(num)/self.conf.n_clients for num in self.conf.arch_info["num_clients_per_model"].split(":")]
        proportion = rate

        archs = self.conf.arch_info["worker"]
        arch_idx = torch.multinomial(torch.tensor(proportion), num_samples=self.conf.n_clients,
                                     replacement=True).tolist()

        self.clientid2arch = dict(
            (
                client_id,
                archs[arch_idx[client_id - 1]]
            )
            for client_id in range(1, 1 + self.conf.n_clients)
        )
        self.conf.clientid2arch = self.clientid2arch

        self.clientid2archindex = dict(
            (
                client_id,
                arch_idx[client_id - 1]
            )
            for client_id in range(1, 1 + self.conf.n_clients)
        )
        return self.clientid2archindex

    def _activate_selected_clients(
            self, selected_client_ids, comm_round, list_of_local_n_epochs
    ):
        # Activate the selected clients:
        # the first row indicates the client id,
        # the second row indicates the current_comm_round,
        # the third row indicates the expected local_n_epochs
        if len(selected_client_ids) < self.conf.n_participated:
            padding = [0] * (self.conf.n_participated - len(selected_client_ids))
            broadcast_client_ids = copy.deepcopy(selected_client_ids)
            broadcast_client_ids.extend(padding)
            broadcast_client_ids = np.array(broadcast_client_ids)
        else:
            broadcast_client_ids = np.array(selected_client_ids)
        msg_len = 3

        if self.conf.split_mix:
            msg_len += 1
        if self.conf.dynamic:
            msg_len += 1

        activation_msg = torch.zeros((msg_len, len(broadcast_client_ids)))
        activation_msg[0, :] = torch.Tensor(broadcast_client_ids)
        activation_msg[1, :] = comm_round
        activation_msg[2, :] = torch.Tensor(list_of_local_n_epochs)

        if self.conf.split_mix:
            slim_info_len = []
            for client_idx in selected_client_ids:
                slim_info_len.append(self.hetero_agg.get_client_slim(client_idx))
            activation_msg[3, :] = torch.Tensor(slim_info_len)

        if self.conf.dynamic:
            activation_msg[msg_len - 1, :] = torch.Tensor(
                [self.clientid2archindex[client_id] for client_id in broadcast_client_ids]
            )

        dist.broadcast(tensor=activation_msg, src=0)
        self.conf.logger.log(f"Master activated the selected clients.")
        dist.barrier()


    def _send_model_to_selected_clients(self, selected_client_ids, to_send_history=False):
        # the master_model can be large; the client_models can be small and different.
        self.conf.logger.log(f"Master send the models to workers.")

        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            arch = self.clientid2arch[selected_client_id]
            self.client_models[arch] = self.client_models[arch].cpu()

            client_model_state_dict = self.client_models[arch].state_dict()
            flatten_model = TensorBuffer(list(client_model_state_dict.values()))
            dist.send(tensor=flatten_model.buffer, dst=worker_rank)
            self.conf.logger.log(
                f"\tMaster send the current model={arch} to process_id={worker_rank}."
            )
            if self.conf.split_mix:
                slim_ratios, slim_shifts = self.hetero_agg.sample_bases(selected_client_id)
                slim_infos = torch.Tensor([slim_ratios, slim_shifts])
                dist.send(tensor = torch.Tensor(slim_infos), dst = worker_rank)


        dist.barrier()

    def _receive_models_from_selected_clients(self, selected_client_ids):
        self.conf.logger.log(f"Master waits to receive the local models.")
        dist.barrier()

        # init the placeholders to recv the local models from workers.
        flatten_local_models = dict()
        for selected_client_id in selected_client_ids:
            arch = self.clientid2arch[selected_client_id]
            client_tb = TensorBuffer(
                list(self.client_models[arch].state_dict().values())
            )
            client_tb.buffer = torch.zeros_like(client_tb.buffer)
            flatten_local_models[selected_client_id] = client_tb

        # async to receive model from clients.
        reqs = []
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(
                tensor=flatten_local_models[client_id].buffer, src=world_id
            )
            reqs.append(req)

        for req in reqs:
            req.wait()

        dist.barrier()
        self.conf.logger.log(f"Master received all local models.")
        return flatten_local_models

    def _receive_label_counts_from_selected_clients(self, selected_client_ids):
        self.conf.logger.log(f"Master waits to receive the local label counts.")
        dist.barrier()

        # init the placeholders to recv the local models from workers.
        label_counts = dict()
        for selected_client_id in selected_client_ids:
            label_count = torch.zeros(self.conf.num_classes)
            label_counts[selected_client_id] = label_count

        # async to receive model from clients.
        reqs = []
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(
                tensor=label_counts[client_id], src=world_id
            )
            reqs.append(req)

        for req in reqs:
            req.wait()
        dist.barrier()
        self.conf.logger.log(f"Master received all local label counts.")

        MIN_SAMPLES_PER_LABEL = 1
        label_weights = []
        qualified_labels = []

        for label in range(self.conf.num_classes):
            weights = []
            for user in selected_client_ids:
                weights.append(label_counts[user][label])
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
            # uniform
            label_weights.append(np.array(weights) / np.sum(weights))  # obtain p(y)
        label_weights = np.array(label_weights).reshape((self.conf.num_classes, -1))
        print(label_weights)
        return label_weights, qualified_labels

    def _avg_over_archs(self, flatten_local_models):
        if self.conf.low_rank or self.conf.pruning or self.conf.split_mix: # hetero deployment
            self.master_model = self.hetero_agg.aggregate_model(flatten_local_models)
            self.client_models = self.hetero_agg.split_model(self.master_model,self.client_models)
            return self.client_models
        else:
            # FedAvg:
            # average for each arch.
            archs_fedavg_models = {}
            for arch in self.used_client_archs:
                # extract local_models from flatten_local_models.
                _flatten_local_models = {}
                for client_idx, flatten_local_model in flatten_local_models.items():
                    if self.clientid2arch[client_idx] == arch:
                        _flatten_local_models[client_idx] = flatten_local_model

                # average corresponding local models.
                self.conf.logger.log(
                    f"Master uniformly average over {len(_flatten_local_models)} received models ({arch})."
                )
                fedavg_model = self.aggregator.aggregate(
                    master_model=self.master_model,
                    client_models=self.client_models,
                    flatten_local_models=_flatten_local_models,
                    aggregate_fn_name="_s1_federated_average",
                    selected_client_ids=None
                )
                archs_fedavg_models[arch] = fedavg_model
            return archs_fedavg_models

    def _aggregate_model_and_evaluate(self, flatten_local_models, selected_client_ids):
        # uniformly averaged the model before the potential aggregation scheme.
        same_arch = len(self.client_models) == 1

        # uniformly average local models with the same architecture.
        fedavg_models = self._avg_over_archs(flatten_local_models)
        if same_arch:
            fedavg_model = list(fedavg_models.values())[0]
        else:
            fedavg_model = None

        # (smarter) aggregate the model from clients.
        # note that: if conf.fl_aggregate["scheme"] == "federated_average",
        #            then self.aggregator.aggregate_fn = None.
        if self.aggregator.aggregate_fn is not None:
            # evaluate the uniformly averaged model.
            if fedavg_model is not None:
                performance = master_utils.get_avg_perf_on_dataloaders(
                    self.conf,
                    self.coordinator[0],
                    fedavg_model,
                    self.criterion,
                    self.metrics,
                    self.test_loaders,
                    label=f"fedag_test_loader",
                )
            # else:
            #    assert "knowledge_transfer" in self.conf.fl_aggregate["scheme"]
            elif "knowledge_transfer" in self.conf.fl_aggregate["scheme"]:
                performance = None
                for _arch, _fedavg_model in fedavg_models.items():
                    master_utils.get_avg_perf_on_dataloaders(
                        self.conf,
                        self.coordinator[0],
                        _fedavg_model,
                        self.criterion,
                        self.metrics,
                        self.test_loaders,
                        label=f"fedag_test_loader_{_arch}",
                    )

            # aggregate the local models.
            client_models = self.aggregator.aggregate(
                master_model=self.master_model,
                client_models=self.client_models,
                fedavg_model=fedavg_model,
                fedavg_models=fedavg_models,
                flatten_local_models=flatten_local_models,
                selected_client_ids=selected_client_ids
            )
            if self.conf.low_rank or self.conf.pruning or self.conf.split_mix:
                self.master_model.load_state_dict(client_models.state_dict())
                self.client_models = self.hetero_agg.split_model(self.master_model, self.client_models)
            else:
                # here the 'client_models' are updated in-place.
                if same_arch:
                    # here the 'master_model' is updated in-place only for 'same_arch is True'.
                    self.master_model.load_state_dict(
                        list(client_models.values())[0].state_dict()
                    )
                for arch, _client_model in client_models.items():
                    self.client_models[arch].load_state_dict(_client_model.state_dict())

        elif not self.conf.low_rank and not self.conf.pruning and not self.conf.split_mix:
            # update self.master_model in place.
            if same_arch and self.conf.arch_info["master"] == self.conf.arch_info["worker"][0]:
                self.master_model.load_state_dict(fedavg_model.state_dict())

            # update self.client_models in place.
            for arch, _fedavg_model in fedavg_models.items():
                self.client_models[arch].load_state_dict(_fedavg_model.state_dict())

        # evaluate the aggregated model on the test data.
        #milestones = [int(x) for x in self.conf.lr_milestones.split(",")]
        if same_arch:
            # to save time, we do not fix the bn statistics from scratch
            # if self.conf.freeze_bn and self.conf.graph.comm_round >= milestones[0]:
            #     test_model = self.fix_bn_stat(fedavg_model, arch=self.used_client_archs[0])
            # else:
            test_model = fedavg_model
            master_utils.do_validation(
                self.conf,
                self.coordinator[0],
                test_model,
                # self.master_model,
                self.criterion,
                self.metrics,
                self.test_loaders,
                label=f"aggregated_test_loader",
            )

        else:
            for index, (arch, _client_model) in enumerate(self.client_models.items()):
                test_model = copy.deepcopy(_client_model)

                # if self.conf.freeze_bn and self.conf.graph.comm_round >= milestones[0]:
                #     test_model = self.fix_bn_stat(test_model, arch=arch)

                if self.conf.split_mix:
                    if self.conf.pruning:
                        test_model.switch_slim_mode(eval(arch.split('_')[-1]))
                    else:
                        test_model.switch_slim_mode(1.0/float(eval(arch.split('_')[-1])))

                master_utils.do_validation(
                    self.conf,
                    self.coordinator[index],
                    test_model,
                    self.criterion,
                    self.metrics,
                    self.test_loaders,
                    label=f"aggregated_test_loader_{arch}",
                )
        # torch.cuda.empty_cache()
        # return performance.dictionary['top1']


    def fix_bn_stat(self, model, arch):
        with torch.no_grad():

            self.test_models[arch].load_state_dict(model.state_dict(), strict=False)
            self.test_models[arch] = self.test_models[arch].cuda()
            self.test_models[arch].train(True)

            # total_step = len(self.data_loader) / self.conf.n_clients * self.conf.n_participated
            for _input, _target in self.data_loader:
                input = _input.cuda()
                self.test_models[arch](input)

        return self.test_models[arch]


    def _check_early_stopping(self):
        meet_flag = False

        # consider both of target_perf and early_stopping
        if self.conf.target_perf is not None:
            assert 100 >= self.conf.target_perf > 0

            # meet the target perf.
            if (
                    self.coordinator[0].key_metric.cur_perf is not None
                    and self.coordinator[0].key_metric.cur_perf > self.conf.target_perf
            ):
                self.conf.logger.log("Master early stopping: meet target perf.")
                self.conf.meet_target = True
                meet_flag = True
            # or has no progress and early stop it.
            elif self.early_stopping_tracker(self.coordinator[0].key_metric.cur_perf):
                self.conf.logger.log(
                    "Master early stopping: not meet target perf but has no patience."
                )
                meet_flag = True
        # only consider the early stopping.
        else:
            if self.early_stopping_tracker(self.coordinator[0].key_metric.cur_perf):
                meet_flag = True

        if meet_flag:
            # we perform the early-stopping check:
            # (1) before the local training and (2) after the update of the comm_round.
            _comm_round = self.conf.graph.comm_round - 1
            self.conf.graph.comm_round = -1
            self._finishing(_comm_round)

    def _finishing(self, _comm_round=None):
        self.conf.logger.save_json()
        self.conf.logger.log(f"Master finished the federated learning.")
        self.conf.is_finished = True
        self.conf.finished_comm = _comm_round
        checkpoint.save_arguments(self.conf)
        os.system(f"echo {self.conf.checkpoint_root} >> {self.conf.job_id}")


    def get_label_split(self):
        label_split = {0: torch.arange(0, self.conf.num_classes)}
        for client_idx, stat_infos in self.data_partitioner.targets_of_partitions.items():
            unique_elements = [stat_info[0] for stat_info in stat_infos]
            label_split[client_idx + 1] = torch.tensor(unique_elements, dtype=torch.int64)

        return label_split


def get_n_local_epoch(conf, n_participated):
    if conf.min_local_epochs is None:
        return [conf.local_n_epochs] * n_participated
    else:
        # here we only consider to (uniformly) randomly sample the local epochs.
        assert conf.min_local_epochs > 1.0
        random_local_n_epochs = conf.random_state.uniform(
            low=conf.min_local_epochs, high=conf.local_n_epochs, size=n_participated
        )
        return random_local_n_epochs
