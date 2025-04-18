## Requirements:
The environment requires to build pytorch from source to support MPI backend.

The code and environment are based on FedDF: https://github.com/epfml/federated-learning-public-code.

MPI is not necessary, use gloo as communication backend.
## Parameters
- global_rate(k): the largest participant model shrink rate(For HeteroFL)
- num_clients_per_model(n): the number of clients for each model
- non_iid_alpha(a): the non-iid degree, setting 0 as iid
- participation_ratio(r): participant ratio each communication round
- data(d): dataset for training
- worker_arch(c): heterogeneous arch for clients, plus "_" means compression parameter
- num_clients_per_model(n): split by ":", the number of clients per group
- low_rank: setting true to use FedHM
- pruning: setting true to use HeteroFL 
- split_mix: setting true to use Split-Mix
- dynamic: setting true to dynamically vary the clients computation power
- freeze_bn: setting true to turn off tracking_status(For sBN)
- need_scaler: setting true to use scaler(For HeteroFL and Split-Mix)
## Run Experiments
### CIFAR-10 with ResNet-18 
The setup of FedAvg for resnet-18 with cifar10:
```
./run_lr.sh -m fedavg -g 0,1,2,3 -d cifar10 -s resnet18 -c resnet18_0.35 -a 0 -r 0.5
```

The setup of FedHM for resnet-18 with cifar10:
```
./run_lr.sh -m lowrank -g 0,1,2,3 -d cifar10 -s resnet18_1 -c resnet18_1:resnet18_2:resnet18_4:resnet18_8 -a 0 -r 0.5 -n 5:5:5:5
```

The setup of Split-Mix for resnet-18 with cifar10:
```
./run_lr.sh -m split_mix -g 2,3 -d cifar10 -s ensresnet18_3.15 -c ensresnet18_3.15:ensresnet18_1.05:ensresnet18_0.7:ensresnet18_0.35 -a 0.5 -r 0.5 -n 5:5:5:5
```

The setup of HeteroFL for resnet-18 with cifar10:
```
./run_lr.sh -m pruning -g 0,1,2,3 -d cifar10 -s resnet18_1 -c resnet18_1:resnet18_0.64:resnet18_0.5:resnet18_0.35 -a 0 -r 0.5 -n 5:5:5:5 -k 1
```

```
./run_cifar100.sh -m pruning -g 0,1,2,3 -d cifar100 -s resnet34_1 -c resnet34_1:resnet34_0.64:resnet34_0.5:resnet34_0.4 -a 0.1 -r 0.5 -n 5:5:5:5
```

### CIFAR-10 with VIT
The setup of the FedAvg for vit-small with cifar10:
```
./run.sh -m fedavg -g 0,1,2,3 -d cifar10 -s vit -c vit -a 0.5 -r 0.2
```

The setup of FedHM for vit-small with cifar10:
```
./run.sh -m lowrank -g 2,0,1,3 -d cifar10 -s vit -c vit:vit_2:vit_4:vit_8 -a 0.01 -r 0.2 -n 5:5:5:5
```