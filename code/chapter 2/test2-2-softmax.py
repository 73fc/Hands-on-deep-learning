import sys
sys.path.append(r'E:\算法刷题\动手学深度学习')
import d2lzh as d2l
from mxnet  import autograd, nd

#-----------------------------获取数据---------------------------------------
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#-----------------------------构建模型---------------------------------------
##初始化参数
num_inputs = 784
num_outputs = 10

W = nd.random.normal( scale=0.01, shape=(num_inputs, num_outputs))
b = nd.random.normal( scale=0.01, shape=(num_outputs))
W.attach_grad()
b.attach_grad()


### 学习nd.sum()
X = nd.array( [[1,1,1],[3,3,3]])
X.sum(axis=0,keepdims=False) ##keepdims 决定合并以后维度是否下降
X.sum(axis=(0,1),keepdims=True) #求和的维度可以扩展
X.sum(axis=1,keepdims=True)

##实现softmax函数
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1,keepdims=True)
    return X_exp/partition

### 测试 softmax
X = nd.random.normal(shape=(2,5))
X_prob = softmax(X)
X_prob, X_prob.sum(axis=1)

##定义模型
def net(X):                       
    return softmax(nd.dot(X.reshape((-1,num_inputs)),W) + b) # 值-1的维度由系统推算出

##定义损失函数
### 学习 pick函数,从列表中选择出指定项
y_hat = nd.array( [[0.1,0.3,0.6],[0.3,0.2,0.5] ])
y = nd.array([0,1],dtype='int32')
nd.pick(y_hat,y)

### 交叉熵损失函数
def cross_entropy(y_hat,y):
    return 0 - nd.log(nd.pick(y_hat,y))

## 定义准确率函数
def accuracy(y_hat,y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

accuracy(y_hat,y)

def evaluate_accuracy(data_iter,net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y ).sum().asscalar()
        n += y.size
    return acc_sum/n



## 训练模型
num_epochs = 7
lr = 0.01

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
            params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum ,train_acc_sum  = 0.0, 0.0
        n = 0
        for X,y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat,y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(params,lr, batch_size)
            else:
                trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter,net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'%(epoch + 1, train_l_sum / n, train_acc_sum/n, test_acc))

train_ch3( net, train_iter, test_iter, cross_entropy, num_epochs, batch_size,[W,b],lr)


for X,y in test_iter:
    break

true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
title = [ true + '\n' + pred for true, pred in zip(true_labels, pred_labels) ]

d2l.show_fashion_mnist(X[0:9],title[0:9])

#-------------------------------------练习-----------------------------
##1. 直接按softmax的数学定义来实现会有什么问题
import math
math.exp(50)
math.exp(100)
### 答： 数据量过大时会导致数据溢出
##2.cross_entropy函数按数学定义实现会有什么问题
##答： 输入数据可能会有负数
##3. 有什么办法解决以上两个问题
##答：数据标准化，规范化

#-------------------------------------gloun实现-----------------------------
import sys
sys.path.append(r'E:\算法刷题\动手学深度学习')
import d2lzh as d2l
from mxnet import gluon,init
from mxnet.gluon import loss as gloss, nn
## 导入数据
batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

## 定义模型与初始化
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

##softmax和交叉熵函数
loss = gloss.SoftmaxCrossEntropyLoss()

##优化算法
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03} )

##训练模型
num_epochs = 7
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,trainer)

