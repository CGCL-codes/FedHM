import re,os
import numpy as np

def unique(lst):
    return dict(zip(*np.unique(lst, return_counts=True)))

data_dir = './data/checkpoint/cifar10/'
#lowrank_file = 'resnet18_1:resnet18_4:resnet18_8:resnet18_18/lowrank_alpha0_ratio0.5'
structure = 'resnet18_1:resnet18_0.5:resnet18_0.35:resnet18_0.25/pruning_alpha0_ratio0.5'
root = os.path.join(data_dir,structure)

data_list = os.listdir(root)

f = open(os.path.join(root,data_list[-1],'0','log.txt'))
time_pattern = '\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2}:\d{1,2}'

pattern = time_pattern + '\s*'+ "Master send the current model=[a-zA-Z1-9_]*(\d+\.\d+|\d+) to process_id=\d+\.\s*"
pattern = re.compile(pattern)

f_data = f.read()
res = re.findall(pattern,f_data)
groups = [res[i*10:i*10+10] for i in range(200)]

to_draw = []
for i,grp in enumerate(groups):
    data = [float(num) for num in grp]
    freq = unique(data)
    if 1 not in freq:
        print(i,freq)
        to_draw.append(i)

to_draw = np.array(to_draw)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

def draw_movement(entry, length):


    data_file = os.path.join(root, entry, '0', 'log-1.json')
    datas = json.load( open(data_file))
    max_acc = 0
    yi = []
    for i, data in enumerate(datas):
        if (i % 8) == 0 and len(yi) < length:
            yi.append(data['top1'])

    yi = np.array(yi)

    x = np.array(list(range(length)) )
    sns.lineplot(
        x=x,#[25:150],
        y=yi,#[25:150].astype(float)
        # legend='brief',
        # ci="sd",
    )
    plt.scatter(to_draw,yi[to_draw],color='red',alpha=0.5,linewidths=0.5)
    min_acc = 70
    max_acc = np.max([max_acc, np.max(yi)]) + 1
    plt.ylim(min_acc, max_acc)

    plt.gcf()
    plt.grid()

    plt.ylabel('Top-1 test accuracy',fontsize=20)
    plt.xlabel('# of communication rounds',fontsize=20)
    plt.show()

draw_movement(data_list[-1],200)
