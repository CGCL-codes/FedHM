#!/usr/bin/env bash

usage="args:[-c,-s,-m,-d,-a,-r,-g]"
num_clients_per_model=0
model=""
dataset=""
alpha=""
ratio=""
gpu=0
global_rate=1
port=52022
GETOPT_ARGS=`getopt -o s:c:m:n:d:a:r:g:k:h:p -- "$@"`
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
    -h) echo $usage; break ;;
    -p) port=$2; shift 2;;
    --) break ;;
  esac

done


if [[ -z $model || -z worker_arch || -z $dataset || -z $alpha || -z $ratio ]]; then
  echo "You should give all the arguments:"
  echo "model=$model, dataset=$dataset, alpha=$alpha, ratio=$ratio, worker_arch=$worker_arch"
  exit 0
fi

arch=$worker_arch

suppoert_models=("FedAvg" "LowRank" "Pruning" "FedProx" "split_mix" "split_svd")
if ! (echo "${suppoert_models[@]}" | grep -wi -q "$model"); then
  echo "Unsupported Model: $model"
  echo "We now support: ${suppoert_models[*]}"
  exit 0
fi

buffer_length=0
n_clients=20
n_comm_rounds=100
num_workers=8
n_local_epochs=10

partition_type='non_iid_dirichlet'
if [ $alpha == 0 ]; then
  partition_type='origin'
fi

freeze_bn=False
need_scaler=False
low_rank=False
pruning=False
dynamic=False
warmup=0
lr_milestones=70,90
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
  #self_distillation=0.1
  #dynamic=True
  low_rank=True
  loss_temperature=5
  freeze_bn=True
elif echo "Pruning" | grep -wi -q "$model"; then
  #dynamic=True
  pruning=True
  need_scaler=True
  freeze_bn=True
fi

experiment="$model"_alpha"$alpha"_ratio"$ratio"_warmup"$warmup"_dynamic"$dynamic"
group_norm_num_groups=0

export "CUDA_VISIBLE_DEVICES=$gpu"
list_of_manual_seed=(0)

val_dataset=0
val_ratio=0

use_lmdb_data=False
if [ $dataset == "tiny-imagenet" ]; then
   use_lmdb_data=True
fi

optimizer="sgd"
learning_rate=0.1
weight_decay=1e-4

fl_aggregate="scheme=federated_average"
time=$(date "+%Y%m%d%H%M%S")
err_file=errors/"$time"_"$arch"_"$experiment"_"$dataset".log

for((num=0;num<1;num++));
do
  echo ============= Round $num Start =============
  tmp_file=results/"$time"+"$experiment"_"$dataset"_round"$num".txt
  python -W ignore run_gloo.py \
      --arch $arch --complex_arch master=$master_arch,worker=$worker_arch,num_clients_per_model=$num_clients_per_model \
      --pin_memory True --batch_size 64 --num_workers $num_workers \
      --partition_data  $partition_type --train_data_ratio 1 --val_data_ratio $val_ratio \
      --val_dataset $val_dataset --use_lmdb_data $use_lmdb_data\
      --n_clients $n_clients --n_comm_rounds $n_comm_rounds --local_n_epochs $n_local_epochs  --world_conf 0,0,1,1,100 --on_cuda True \
      --optimizer $optimizer --lr $learning_rate --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
      --lr_scheduler MultiStepLR --lr_decay 0.1  --lr_milestones $lr_milestones\
      --weight_decay $weight_decay --use_nesterov False --momentum_factor 0.9 \
      --low_rank $low_rank --pruning $pruning --dynamic $dynamic --freeze_bn $freeze_bn --need_scaler $need_scaler --warmup $warmup\
      --track_time True --display_tracked_time True --python_path  $(which python) --hostfile hostfile \
      --manual_seed ${list_of_manual_seed[$num]} --pn_normalize True --same_seed_process True \
      --experiment $experiment --data $dataset --non_iid_alpha $alpha --participation_ratio $ratio \
      --loss_temperature $loss_temperature \
      --group_norm_num_groups $group_norm_num_groups \
      --unit $unit --split_mix $split_mix\
      --fl_aggregate $fl_aggregate \
      --self_distillation $self_distillation \
      --global_rate $global_rate \
      --port $port \
      --num_classes 100 
      #2>$err_file
      #1>$tmp_file \

  echo ============= Round $num Result =============
  echo "============= Round $num Result ============="

  grep "performance" $tmp_file | tail -n 5
  grep "performance" $tmp_file | tail -n 5
done


