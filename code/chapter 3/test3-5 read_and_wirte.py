from mxnet import nd
from mxnet.gluon import nn

#------------------------------读写 NDrrays---------------------------------
# 写nd变量
x = nd.ones(3)
nd.save('x',x)
# 读nd变量
x2 = nd.load('x')
x2 

# 读写多变量（一列表） 
y = nd.zeros(4)
nd.save('xy',[x,y])
x2,y2 = nd.load('xy')
x2,y2

# 读写字典
mydict = {'x':x,'y':y}
nd.save('mydict',mydict)
mydict2 = nd.load('mydict')
mydict2

#---------------------------读写Gluon模型参数-----------------------------------
#创建模型并初始化
class MLP(nn.Block):
    def __init__(self,**kwargs):
        super(MLP,self).__init__(**kwargs)
        self.hidden = nn.Dense(256,activation='relu')
        self.output = nn.Dense(3)
    
    def forward(self,x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
x = nd.random.uniform(shape=(2,20))
y = net(x)

#存储
filename = 'mlp.params'
net.save_parameters(filename)

#读取模型参数
net2 = MLP()
net2.load_parameters(filename)
y2 = net2(x)
y2 == y

#---------------------------练习-----------------------------------
'''
提问：即使无需部署，存储模型参数还有哪些好处？
答：有利于保持训练中的中间结果，并随时继续训练
'''

