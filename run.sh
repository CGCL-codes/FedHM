#!/usr/bin/env bash

usage="args:[-c,-s,-m,-d,-a,-r,-g]"
num_clients_per_model="20"
model=""
dataset=""
alpha=""
ratio=""
gpu=0
global_rate=1
port=50022
GETOPT_ARGS=`getopt -o s:c:m:n:d:a:r:g:k:p:h -- "$@"`
eval set -- "$GETOPT_ARGS"

while [ -n "$1" ]
do
  case "$1" in
    -s) master_arch=$2; shift 2;;
    -c) worker_arch=$2; shift 2;;
    -m) model=$2; shift 2;;
    -n) num_clients_per_model=$2; shift 2;;
    -d) dataset=$2; shift 2;;
    -a) alpha=$2; shift 2;;
    -r) ratio=$2; shift 2;;
    -g) gpu=$2; shift 2;;
    -k) global_rate=$2; shift 2;;
    -p) port=$2; shift 2;;
    -h) echo $usage; break ;;
    --) break ;;
  esac
done

if [[ -z $model || -z $worker_arch || -z $dataset || -z $alpha || -z $ratio ]]; then
  echo "You should give all the arguments:"
  echo "model=$model, dataset=$dataset, alpha=$alpha, ratio=$ratio, worker_arch=$worker_arch, master_arch=$master_arch"
  exit 0
fi


suppoert_models=("FedAvg" "LowRank" "Pruning" "FedProx" "split_mix" "split_svd")
if ! (echo "${suppoert_models[@]}" | grep -wi -q "$model"); then
  echo "Unsupported Method: $model"
  echo "We now support: ${suppoert_models[*]}"
  exit 0
fi

arch=$worker_arch
buffer_length=0
n_clients=20
n_comm_rounds=60
num_workers=8
n_local_epochs=1

partition_type='non_iid_dirichlet'
if [ $alpha == 0 ]; then
  partition_type='origin'
fi

freeze_bn=False
need_scaler=False
low_rank=False
pruning=False
dynamic=False
warmup_rounds=0
loss_temperature=0
unit=False
self_distillation=0
split_mix=False

if echo "split_mix" | grep -wi -q "$model"; then
    split_mix=True
    need_scaler=True
    freeze_bn=True
    pruning=True
elif echo "split_svd" | grep -wi -q "$model"; then
    split_mix=True
    freeze_bn=True
    low_rank=True
elif echo "LowRank" | grep -wi -q "$model"; then
  low_rank=True
  loss_temperature=5
  freeze_bn=True
elif echo "Pruning" | grep -wi -q "$model"; then
  dynamic=True
  pruning=True
  need_scaler=True
  freeze_bn=True
fi

num_classes=10
if echo "cifar100" | grep -wi -q "$dataset"; then
  num_classes=100
elif echo "cifar10" | grep -wi -q "$dataset"; then
  num_classes=10
elif echo "tiny" | grep -wi -q "$dataset"; then
  num_classes=200
fi

experiment="$model"_alpha"$alpha"_ratio"$ratio"_warmup"$warmup"_dynamic"$dynamic"
group_norm_num_groups=0

export "CUDA_VISIBLE_DEVICES=$gpu"
echo "using gpu:$gpu for training"
list_of_manual_seed=(0)

val_dataset=0
val_ratio=0

optimizer="adam"
learning_rate=1e-4
weight_decay=0

fl_aggregate="scheme=federated_average"
time=$(date "+%Y%m%d%H%M%S")

for((num=0;num<1;num++));
do
  echo ============= Round $num Start =============
  python -W ignore run_gloo.py \
      --arch $arch --complex_arch master=$master_arch,worker=$worker_arch,num_clients_per_model=$num_clients_per_model \
      --pin_memory True --batch_size 64 \
      --partition_data  $partition_type --train_data_ratio 1 --val_data_ratio $val_ratio \
      --val_dataset $val_dataset \
      --n_clients $n_clients --n_comm_rounds $n_comm_rounds --local_n_epochs $n_local_epochs  --world_conf 0,0,1,1,100 --on_cuda True \
      --optimizer $optimizer --lr $learning_rate --lr_warmup False \
      --lr_scheduler WarmupCosine \
      --weight_decay $weight_decay --use_nesterov False\
      --low_rank $low_rank --pruning $pruning --dynamic $dynamic --freeze_bn $freeze_bn --need_scaler $need_scaler --warmup_rounds $warmup_rounds\
      --track_time True --display_tracked_time True --python_path  $(which python) --hostfile hostfile \
      --manual_seed ${list_of_manual_seed[$num]} --pn_normalize True --same_seed_process True \
      --experiment $experiment --data $dataset --non_iid_alpha $alpha --participation_ratio $ratio \
      --group_norm_num_groups $group_norm_num_groups \
      --unit $unit --split_mix $split_mix\
      --fl_aggregate $fl_aggregate \
      --self_distillation $self_distillation \
      --global_rate $global_rate \
      --port $port --num_classes $num_classes\

  echo ============= Round $num End =============
done


