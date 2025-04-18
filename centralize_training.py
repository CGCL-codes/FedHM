import torch
from pcode.models.resnet import ResNet_imagenet
from pcode.models.lowrank_resnet import HybridResNet,LowRankBasicBlockConv1x1,LowRankBottleneck,FullRankBasicBlock,LowRankBasicBlock,LowRankBottleneckConv1x1
from pcode.datasets.prepare_data import get_cifar
from torch.utils.data import DataLoader
from test_lowrank import svd_combine_model,svd_split_model,init_svd,init_pruning_index,pruning_split_model,distill_large_model
import torch.nn as nn
from pcode.create_metrics import obtain_accuracy, AverageMeter
import numpy,random,os
from pcode.utils.param_parser import str2bool
from pcode.utils.flops_counter import get_model_complexity_info
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--epochs",type=int, default=200)
parser.add_argument("--warmup",type=int, default=20)
parser.add_argument("--square",type=str2bool, default=False)
parser.add_argument("--rank",type=float,default=4)
parser.add_argument("--gpu",type=str,default='0')
parser.add_argument("--pruning",type=str2bool, default=False)
parser.add_argument("--scaler_rate",type=float,default=0.5)
args = parser.parse_args()

torch.cuda.set_device(torch.device("cuda:"+args.gpu))
def param_counter(model):
    num_params = 0
    for param_index, (param_name, param) in enumerate(model.named_parameters()):
        num_params += param.numel()
    return num_params / 1e6
def init_seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(0)
    numpy.random.seed(0)

def param_counter(model):
    num_params = 0
    for param_index, (param_name, param) in enumerate(model.named_parameters()):
        num_params += param.numel()
    return num_params / 1e6

def get_loaders(dataset):
    train_dataset = get_cifar(name=dataset,
                              root=os.path.join('./data/',dataset),
                              split='train',
                              target_transform=None,
                              download=False)

    test_dataset = get_cifar(
        name=dataset,
        root=os.path.join('./data/',dataset),
        split='test',
        target_transform=None,
        download=False
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=64,
                              shuffle=False, num_workers=4, pin_memory=False)

    test_loader = DataLoader(dataset=test_dataset, batch_size=64,
                             shuffle=False, num_workers=4, pin_memory=False)
    return train_loader,test_loader

init_seed()

resnet_size = 34
rank_factor = int(args.rank)
fullrank_model = ResNet_imagenet("tiny-imagenet",resnet_size,freeze_bn=False,need_scaler=True).cuda()
#fullrank_model = EfficientNetB0(10).cuda()
# fullrank_model = DenseNet("cifar100", 100, 12, True, 0.5, 0,
#             need_scaler=False,
#             scaler_rate=1,
#             track_running_stats = True).cuda()

#print("============> FullRank Model info: num params: {}".format(param_counter(fullrank_model)))

# for index, (param_name, param) in enumerate(fullrank_model.state_dict().items()):
#     print(index, param_name)
fullrank_macs, fullrank_params = get_model_complexity_info(fullrank_model, (3, 64, 64), as_strings=True,
                                                 print_per_layer_stat=True, verbose=False)
print(fullrank_macs)
print(param_counter(fullrank_model))
if args.pruning:
    # lowrank_model = DenseNet("cifar100", 100, 12, True, 0.5, 0,
    #         need_scaler=False,
    #         scaler_rate=args.scaler_rate,
    #         track_running_stats = True).cuda()
    lowrank_model = ResNet_imagenet("tiny-imagenet", resnet_size, freeze_bn=False, pruning=True, scaler_rate= args.scaler_rate,
                                    need_scaler=True).cuda()

    init_seed()
    print("============> Pruning Model info: num params: {}".format(param_counter(lowrank_model)))
    arch_name = 'resnet34_' + str(args.scaler_rate)
    #index_map, arch = init_pruning_index(fullrank_model, arch=arch_name)
    #lowrank_model = pruning_split_model(index_map, arch, lowrank_model,fullrank_model)

else:
    # lowrank_model = DenseNet("cifar100", 100, 12, True, 0.5, 0,
    #         need_scaler=False,
    #         rank_factor=rank_factor,
    #         track_running_stats = True).cuda()
    #lowrank_model = EfficientNetB0(num_classes=10,factor=rank_factor).cuda()
    lowrank_model = HybridResNet(LowRankBasicBlock, FullRankBasicBlock,rank_factor=rank_factor, layers=[2,2,2,2],
                                 num_classes=200,track_running_stats=True,square=args.square).cuda()

    #print(sum(p.numel() for p in lowrank_model.state_dict().values()) / 1e6)
    init_seed()

    #index_map, arch = init_svd(fullrank_model, lowrank_model, arch='resnet34_' + str(rank_factor))
    #lowrank_model = svd_split_model(fullrank_model, lowrank_model, index_map, arch, args.square)
lowrank_macs, lowrank_params = get_model_complexity_info(lowrank_model, (3, 224, 224), as_strings=True,
                                    print_per_layer_stat=True, verbose=False)
#print(sum(p.numel() for p in lowrank_model.state_dict().values()) / 1e6)
print(param_counter(lowrank_model))
print(lowrank_macs)
print("============> Lowrank Model info: num params: {},macs:{}".format(param_counter(lowrank_model),lowrank_macs))
train_loader,test_loader = get_loaders('cifar100')
#print(param_counter(fullrank_model))
# lowrank_macs, lowrank_params = get_model_complexity_info(lowrank_model, (3, 32, 32), as_strings=True,
#                                                print_per_layer_stat=False, verbose=False)
# fullrank_macs, fullrank_params = get_model_complexity_info(fullrank_model, (3, 32, 32), as_strings=True,
#                                                 print_per_layer_stat=False, verbose=True)
# pruning_macs, pruning_params = get_model_complexity_info(pruning_model, (3, 32, 32), as_strings=True,
#                                                print_per_layer_stat=False, verbose=True)
#print("============> Lowrank Model info: num params: {}, Macs: {}".format(param_counter(lowrank_model), lowrank_macs))
# #print("============> Pruning Model info: num params: {}, Macs: {}".format(param_counter(pruning_model), pruning_macs))
# print("============> Fullrank Model info: num params: {}, Macs: {}".format(param_counter(fullrank_model), fullrank_macs))


#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
criterion = nn.CrossEntropyLoss()

from tqdm import tqdm

def apply_frobdecay(name, module, skiplist=[]):
    if hasattr(module, 'frobgrad'):
        return not any(name[-len(entry):] == entry for entry in skiplist)
    return False


def _add_frob_grad(model,weight_decay = 1e-4):

    for i, module in enumerate(module for name, module in model.named_modules()
                               if apply_frobdecay(name, module, skiplist=[])):
        module.frobgrad(coef=weight_decay)

def train(model, loader):
    model.train()
    model.requires_grad_(True)

    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()

    for data, target in tqdm(loader):
        data, target = data.cuda(),target.cuda()

        _,y = model(data)

        loss = criterion(y, target)

        optimizer.zero_grad()
        loss.backward()
        _add_frob_grad(model)
        optimizer.step()

        prec1, prec5 = obtain_accuracy(y, target, topk=(1, 5))
        n = data.size(0)
        losses.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)



    return losses.avg, top1.avg, top5.avg


def test(model, lowrank = True):
    test_model = model
    #
    # test_model.load_state_dict(model.state_dict(), strict=False)
    # test_model.train()
    # with torch.no_grad():
    #     step = 0
    #     for data, target in train_loader:
    #         data, target = data.cuda(),target.cuda()
    #         _,y  = test_model(data)
    #         step += 1
    #         if step >= 100:
    #             break


    test_model.eval()
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.cuda(),target.cuda()
            _,y  = test_model(data)

            loss = criterion(y, target)

            prec1, prec5 = obtain_accuracy(y, target, topk=(1, 5))
            n = data.size(0)
            losses.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        return losses.avg, top1.avg, top5.avg

warmup_epoch = args.warmup

best_acc = 0
train_model = fullrank_model
split_acc,split_epoch = 0,0
lr = 0.1

optimizer = torch.optim.SGD(train_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

for epoch in range(1, args.epochs + 1):
    if epoch == warmup_epoch:
        #lowrank_model = distill_large_model(fullrank_model,lowrank_model,distill_loader)
        # print("#####distill test########")
        # loss, top1, _ = test(lowrank_model, lowrank=True)
        # if top1 > split_acc:
        #     split_acc = top1
        #     split_epoch = epoch
        # print(f'epoch:{epoch},test loss:{loss:.4f}, top1:{top1:.4f}')
        #print(f"best top1:{split_acc:.4f}, best epoch:{split_epoch}")

        train_model = lowrank_model
        no_decay = ['conv1_u', 'conv1_v', 'conv2_u', 'conv2_v']
        params = [
            {
                "params": [value],
                "name": key,
                "weight_decay": 1e-4 if not any(nd in key for nd in no_decay) else 0.0,
                "param_size": value.size(),
                "nelement": value.nelement(),
            }
            for key, value in train_model.named_parameters()
        ]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)

    train(train_model, train_loader)

    if epoch < warmup_epoch:
        print("#####fullrank test#######")
    else:
        print("#####lowrank test########")
    loss, top1, _ = test(train_model,lowrank=True)
    if top1 > best_acc:
        best_acc = top1
    print(f'epoch:{epoch},test loss:{loss:.4f}, top1:{top1:.4f},best top1:{best_acc:.4f}')

    if epoch < warmup_epoch:
        print("#####split test########")
        if args.pruning:
            lowrank_model = pruning_split_model(index_map,arch,lowrank_model,fullrank_model)
        else:
            lowrank_model = svd_split_model(train_model, lowrank_model, index_map, arch,args.square)

        loss, top1, _ = test(lowrank_model, lowrank=True)
        if top1 > split_acc:
            split_acc = top1
            split_epoch = epoch
        print(f'epoch:{epoch},test loss:{loss:.4f}, top1:{top1:.4f}')
        print(f"best top1:{split_acc:.4f}, best epoch:{split_epoch}")
        # fullrank_model = svd_combine_model(fullrank_model,lowrank_model,index_map,arch)
        # print("#####combine test########")
        # loss, top1, _ = test(fullrank_model,False)
        # print(f'epoch:{epoch},test loss:{loss:.4f}, top1:{top1:.4f}')


    if epoch < args.epochs / 2:
        pass
    elif epoch < args.epochs * (3/4):
        for group in optimizer.param_groups:
            group['lr'] = lr / 10.0
    else:
        for group in optimizer.param_groups:
            group['lr'] = lr / 100.0



