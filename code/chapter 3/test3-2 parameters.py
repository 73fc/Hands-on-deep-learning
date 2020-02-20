from mxnet import init, nd
from mxnet.gluon import nn

#-------------------------------构建模型-------------------------
net = nn.Sequential()
net.add(
    nn.Dense(256,activation='relu'),
    nn.Dense(10)
)
net.initialize()


x = nd.random.normal(shape=(2,20))
y = net(x)

#---------------------------获取及查看信息-----------------------
##查看参数信息
net[0].params
type(net.params)

## 查看参数数据
net[0].weight.data()
net[1].bias.data()

## 查看梯度
net[0].weight.grad()

## 获取net的所有参数的
net.collect_params()

## 查看collect_params 返回的类型
type(net.collect_params())

## 用正则表达式来匹配参数名
net.collect_params('.*weight')

#------------------------------初始化模型参数---------------------------
## 使用初始化函数
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True) #再次初始化需要 设置 force_reinit=True
net[0].weight.data()[0]

## 使用常数初始化
net.initialize(init=init.Constant(1),force_reinit=True)
net[0].weight.data()[0]

## 对某个指定参数用指定方法初始化
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[0].weight.data()[0]

#-------------------------------自定义初始化方法---------------------------
## 继承 init.Initialize类，重写 _init_weight函数
class MyInit(init.Initializer):
    def _init_weight(self,name,data):
        print('Init',name,data.shape)
        data[:] = nd.random.uniform(low=-10,high=10, shape=data.shape)
        data *= data.abs() >= 5

net.initialize(MyInit(),force_reinit=True)
net[0].weight.data()[0]

## 直接对参数值做修改
net[0].weight.set_data(net[0].weight.data() + 1)
net[0].weight.data()[0]

#-------------------------------共享模型参数----------------------------
## 1. 在forward中多次调用同一个层
## 2. 构建模型时使用特定的参数，如下

net = nn.Sequential()
shared_Dense = nn.Dense(8,activation='relu')
net.add( nn.Dense(8,activation='relu'),
        shared_Dense,
        nn.Dense(8,params=shared_Dense.params),
        nn.Dense(3))
    
net.initialize()
net.collect_params()
x = nd.random.uniform(shape=(2,20))
net(x)
net.collect_params()
#-------------------------------练习----------------------------
'''
1. 查看文档，了解不同的初始化方法
2.尝试在net.initialize()之后， net(x)之前查看模型参数的形状
答： 没有具体运行之前，net中各个层次彼此不知道对方的存在，所有每一层只确定输出的维度，不确定输入的维度
且一旦一种类型的参数输入后形状就已经确定了，后续只能输入模型所接受的维度的数据
3. 构造一个含共享参数层的MLP并训练，训练时观察每一层的模型参数和梯度
'''

from mxnet import autograd, nd
from mxnet.gluon import data as gdata
#--------------------初始化数据-----------------------
num_inputs = 8
num_example = 100
true_w = [2,-3.4,34,3,5,1,9,11]
true_w = nd.array(true_w)
true_b = 4.2

features = nd.random.normal(scale=1,shape=(num_example,num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01,shape=labels.shape)

##读取数据的辅助函数
batch_size = 10
dataset = gdata.ArrayDataset(features,labels)
data_iter = gdata.DataLoader(dataset,batch_size,shuffle=True)

#------------------构建模型---------------------------
net = nn.Sequential()
shared_Dense = nn.Dense(8,activation='relu')
net.add( nn.Dense(8,activation='relu'),
        shared_Dense,
        nn.Dense(8,params=shared_Dense.params),
        nn.Dense(1))
net.initialize()

##定义损失函数
from mxnet.gluon import loss as gloss
loss = gloss.L2Loss()

## 定义训练器
from mxnet import gluon
trainer =  gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})

## 训练模型
num_epochs = 1
for epoch in range(1,num_epochs + 1):
    for X,y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
        print(net[0].weight.data(), net[0].weight.grad())
        print(net[1].weight.data(),net[1].weight.grad())
        print(net[2].weight.data(),net[2].weight.grad())
        print(net[3].weight.data(),net[3].weight.grad())
    l = loss(net(features), labels)
    print("epoch：%d，loss: %f"%(epoch,l.mean().asnumpy()))

'''
答： 共享之后，共享层的参数值和梯度值都共享了。
'''