from IPython import display
from matplotlib  import pyplot as plt
from mxnet import autograd, nd
import random 
#-----------------------------------------生成数据集-----------------------------
##设置真实数值
num_inputs = 2
num_example = 1000
true_w = nd.array([2, -3.4])
true_b = 4.2
##构建数据集
features = nd.random.normal(scale=1,shape=(num_example,num_inputs))
labels = nd.dot(features,true_w) + true_b
labels += nd.random.normal(scale=0.01,shape=labels.shape)

# ##绘制散点图
# def use_svg_display():
#     display.set_matplotlib_formats('svg')

# def set_figsize(figsize=(3.5,2.5)):
#     use_svg_display()
#     plt.rcParams['figure.figsize'] = figsize 

# set_figsize()
# plt.scatter(features[:,1].asnumpy(),labels.asnumpy(),1)

#-----------------------------------------初始化-----------------------------
## 按辅助函数 批次读取数据
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) 
    for i in range(0, num_examples, batch_size):
        j = nd.array( indices[i:min(i+ batch_size, num_examples)])
        yield features.take(j), labels.take(j)

# ## 测试
batch_size = 10
# for X,y in data_iter(batch_size,features,labels):
#     print(X,y)
#     break

## 初始化参数
w = nd.random.normal(scale=0.01,shape=(num_inputs,1))
b = nd.zeros(shape=(1,))

w.attach_grad()
b.attach_grad()

## 定义模型

def linearg(X,w,b):
    return nd.dot(X,w) + b

## 定义损失函数

def squared_loss(y_hat,y):
    #print(y_hat.shape)
    #print(y.shape)
    return (y_hat - y.reshape(y_hat.shape)) **2 / 2

def sgd(params,lr,batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size 


#-----------------------------------------训练模型-----------------------------

lr = 0.05
num_epochs = 7
net = linearg
loss = squared_loss

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features, labels):
        with autograd.record():
            l = loss(net(X,w,b),y)
    l.backward()
    sgd([w,b],lr,batch_size)
    train_l = loss(net(features,w,b),labels)
    print("epoch %d ,loss %f"%(epoch + 1, train_l.mean().asnumpy()))
    


#-----------------------------------------训练模型-----------------------------
##练习1： 因为前后两者形状不一，一个是（2，）表示是一个一维数组，里面有两个元素，一个是（2，1）表示一个二维数组，总共2行，每行一个元素
#练习2： 使用不同的学习率 
# 0.03 
# epoch 1 ,loss 15.809017
# epoch 2 ,loss 14.460816
# epoch 3 ,loss 13.498562
# epoch 1 ,loss 15.648679
# epoch 2 ,loss 15.033211
# epoch 3 ,loss 14.411322
# epoch 4 ,loss 13.304739
# epoch 5 ,loss 12.422991
# epoch 6 ,loss 11.975589
# epoch 7 ,loss 10.843455
# 0.01
# epoch 1 ,loss 16.575708
# epoch 2 ,loss 16.159794
# epoch 3 ,loss 15.889905
# epoch 4 ,loss 15.488552
# epoch 5 ,loss 15.277861
# epoch 6 ,loss 15.129772
# epoch 7 ,loss 14.907787
# 0.05
# epoch 1 ,loss 15.585565
# epoch 2 ,loss 13.548445
# epoch 3 ,loss 12.980459
# epoch 4 ,loss 10.931391
# epoch 5 ,loss 10.298813
# epoch 6 ,loss 9.801307
# epoch 7 ,loss 9.321486

## 练习3： num_examples 不能被 batch_size 整除， 会导致每个epoch的最后一次梯度下降时训练的样本数少于batch_size

#-----------------------------------------线性回归 gloun版-----------------------------

from mxnet import autograd, nd
##初始化数据
num_inputs = 2
num_example = 1000
true_w = [2,-3.4]
true_b = 4.2

features = nd.random.normal(scale=1,shape=(num_example,num_inputs))
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
labels += nd.random.normal(scale=0.01,shape=labels.shape)

##读取数据的辅助函数
from mxnet.gluon import data as gdata

batch_size = 10
dataset = gdata.ArrayDataset(features,labels)
data_iter = gdata.DataLoader(dataset,batch_size,shuffle=True)


## 定义模型
from mxnet.gluon  import  nn
net = nn.Sequential()

net.add(nn.Dense(1))

##初始化模型参数
from mxnet import init
net.initialize(init.Normal(sigma=0.01))

##定义损失函数
from mxnet.gluon import loss as gloss
loss = gloss.L2Loss()

## 定义训练器
from mxnet import gluon
trainer =  gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})

## 训练模型
num_epochs = 3
for epoch in range(1,num_epochs + 1):
    for X,y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
        net.collect_params()
    l = loss(net(features), labels)
    print("epoch：%d，loss: %f"%(epoch,l.mean().asnumpy()))

## 观察结果
dense = net[0]
true_w, dense.weight.data()
true_b, dense.bias.data()
dense.weight.grad
#----------------------------------------------------练习-------------------------------------------
##练习1：答 因为mean已经把所有数据所产生的损失求和再平均了，所以梯度下降时只需要对每个维度求一次梯度，不需要再对每个数据分布求梯度再相加再平均
##练习2：查资料了解
#练习3：dense.weight.grad