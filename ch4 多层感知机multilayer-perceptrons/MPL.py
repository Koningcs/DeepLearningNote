import torch
from torch import  nn
import torchvision
from torchvision import transforms
from d2l import torch as d2l
from torch.utils import data



batch_size = 256

# 更改成从本地读取数据
trans = [transforms.ToTensor()]
trans = transforms.Compose(trans)
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train = True, transform=trans)
mnist_test = torchvision.datasets.FashionMNIST(root="../data",train=False, transform=trans)
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=0)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=True,num_workers=0)

num_inputs, num_outputs, num_hiddens = 784, 10, 265
W1 = nn.Parameter( torch.randn(num_inputs, num_hiddens, requires_grad = True) )
b1 = nn.Parameter( torch.zeros(num_hiddens, requires_grad = True) )
W2 = nn.Parameter( torch.randn(num_hiddens, num_outputs, requires_grad = True) )
b2 = nn.Parameter( torch.zeros(num_outputs, requires_grad = True) )
params = [W1, b1, W2, b2]

def relu(X):
    # a 是一个形状和X相同的矩阵，里面全是0
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    # 图片拉成矩阵， 固定列数为num_inputs, -1表示计算行数
    X = X.reshape((-1, num_inputs))
    # @ 表示矩阵乘法
    H = relu(X @ W1 + b1)
    return (H @ W2 + b2)

# 交叉熵做损失函数
loss = nn.CrossEntropyLoss()

num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.predict_ch3(net, test_iter)