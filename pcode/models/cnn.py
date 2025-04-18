import torch.nn as nn
import torch
import torch.nn.functional as F

class SimpleCNN_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5,bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5,bias=False)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.classifier = nn.Linear(hidden_dims[-1],output_dim)
        #self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.classifier(x)
        # x = self.fc3(x)
        return x,x

class LowRankCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(LowRankCNN, self).__init__()

        dim1, dim2 = 6* 5, 3 * 5

        self.rank = 1
        #self.padding = 0
        #self.dilation,self.groups,self.stride = 1, 1, 1
        self.conv1_u = nn.Parameter(torch.zeros(self.rank, dim2))
        self.conv1_v = nn.Parameter(torch.zeros(dim1, self.rank))
        #self.conv1_u = nn.Conv2d(3, 1, (1,5),bias=False)
        #self.conv1_v = nn.Conv2d(1, 6, (5,1),bias=False)
        #self.conv1 = nn.Conv2d(3, 6, 5, bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5, bias=False)

        dim1, dim2 = 16* 5 , 6 * 5
        self.conv2_u = nn.Parameter(torch.zeros(self.rank, dim2))
        self.conv2_v = nn.Parameter(torch.zeros(dim1, self.rank))
        # self.conv2_u = nn.Conv2d(6, 1, (1,5),bias=False)
        # self.conv2_v = nn.Conv2d(1,16, (5,1),bias=False)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        # dim1, dim2 = hidden_dims[0],input_dim
        # self.fc1_u = nn.Parameter(torch.zeros(dim1, self.rank))
        # self.fc1_v = nn.Parameter(torch.zeros(self.rank, dim2))
        #self.bias1 = nn.Parameter(torch.zeros(dim2))
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        dim1, dim2 = hidden_dims[1],hidden_dims[0]
        # self.fc2_u = nn.Parameter(torch.zeros(dim1, self.rank))
        # self.fc2_v = nn.Parameter(torch.zeros(self.rank, dim2))
        #self.bias2 = nn.Parameter(torch.zeros(dim2))
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.classifier = nn.Linear(hidden_dims[-1], output_dim)
        #self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        #x = self.conv1(x)
        x = F.conv2d(x,
                       self.conv1_u.T.reshape(3, 5, 1, 1).permute(3, 0, 2, 1),
                       None).contiguous()
        x = F.conv2d(x,
                       self.conv1_v.reshape(6, 5, self.rank, 1).permute(0, 2, 1, 3),
                       None).contiguous()
        x = self.pool(self.relu(x))
        #x = self.conv2(x)
        x = F.conv2d(x,
                       self.conv2_u.T.reshape(6, 5, 1, self.rank).permute(3, 0, 2, 1),
                       None).contiguous()
        # out = self.bn2_u(out)
        x = F.conv2d(x,
                       self.conv2_v.reshape(16, 5, self.rank, 1).permute(0, 2, 1, 3),
                       None).contiguous()
        x = self.pool(self.relu(x))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        #x = F.linear(F.linear(x, self.fc2_v), self.fc2_u)
        x = self.relu(self.fc2(x))
        x = self.classifier(x)
        # x = self.fc3(x)
        return x,x