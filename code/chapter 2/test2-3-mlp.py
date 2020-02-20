import sys
sys.path.append(r'E:/算法刷题/动手学深度学习')
import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import loss as gloss

#------------------------------------------获取数据-------------------------------------------
batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

#------------------------------------------模型设置---------------------------------------------
#初始化参数

num_inputs, num_outputs, num_hiddens = 784,10,256
num_outputs2 = 10
W1 = nd.random.normal(scale=0.01,shape=(num_inputs,num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01,shape=(num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)
W3 = nd.random.normal(scale=0.01,shape=(num_outputs,num_outputs2))
b3 = nd.zeros(num_outputs2)

params = [W1,b1,W2,b2,W3,b3]

for param in params:
    param.attach_grad()

#定义激活函数1`q    
def relu(X):
    return nd.maximum(X,0)

#定义模型
def net(X):
    X = X.reshape((-1,num_inputs))
    H1 = relu( nd.dot(X,W1) + b1)
    H2 = relu(nd.dot(H1,W2) + b2)
    return  nd.dot(H2,W3) + b3

#定义损失函数
loss = gloss.SoftmaxCrossEntropyLoss()

#------------------------------------------模型训练---------------------------------------------
num_epochs,lr = 7,0.5
d2l.train_ch3(net, train_iter,test_iter,loss,num_epochs,batch_size,params,lr)


#------------------------------------------课后练习---------------------------------------------
##1.如何处理以下情况：
#a.数据有缺少
#b.数据有假
# 答: 数据清洗，补全，以及k折交叉验证

##2. 改变num_hiddens的值，看有何影响。


#------------------------------------------Gluon实现---------------------------------------------
import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn

#读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(256)
#构建模型并初始化
net = nn.Sequential()
net.add(nn.Dense(256,activation='relu'),
        nn.Dense(64),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

#定义损失函数
loss = gloss.SoftmaxCrossEntropyLoss()

#训练模型
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5} )
num_epochs = 5
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,trainer)
