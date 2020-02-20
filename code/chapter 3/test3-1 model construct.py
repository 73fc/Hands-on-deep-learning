from mxnet import nd
from mxnet.gluon import nn
#------------------------------------构建模型类--------------------------------------
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP,self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)
    def forward(self,x):
        return self.output(self.hidden(x))

##测试
x = nd.random.normal(shape=(2,20))
net = MLP()
net.initialize()
net(x)



#------------------------------------构建sequential类--------------------------------------
class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential,self).__init__(**kwargs)
    
    def add(self, block):
        self._children[block.name] = block
    
    def forward(self,x):
        for block in self._children.values():
            x = block(x)
        return x

##测试
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(x)

#------------------------------------构建sequential类--------------------------------------
class FancyMLP(nn.Block):
    def __init__(self,**kwargs):
        super(FancyMLP,self).__init__(**kwargs)
        self.rand_weight = self.params.get_constant(
            'rand_weight',nd.random.uniform(shape=(20,20)))
        self.dense = nn.Dense(20,activation='relu')

    def forward(self,x):
        x = self.dense(x)
        x = nd.relu( nd.dot( x, self.rand_weight.data() ) + 3)
        x = self.dense(x)
        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()

net = FancyMLP()
net.initialize()
net(x)

#------------------------------------自建模型与系统模型嵌套类--------------------------------------
class NestMLP(nn.Block):
    def __init__(self,**kwargs):
        super(NestMLP,self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64,activation='relu'),
                    nn.Dense(32,activation='relu'))
        self.dense = nn.Dense(16,activation='relu')
    def forward(self,x):
        return self.dense(self.net(x))

net = nn.Sequential()
net.add(NestMLP(), nn.Dense(20),FancyMLP())
net.initialize()
net(x)


#----------------------------------------------练习--------------------------------------
'''
1. 如果 __init__ 里不对父类进行初始化会怎么样？
答：会有继承的错误， 参数和函数会有变化
AttributeError: 'MLP' object has no attribute '_children'
2.如果在FancyMLP中 去掉 asscalar 会有什么问题？
答： 理论上会有类型不一致的问题（？？） 实际运行没问题。
3.在NestMLP中将Sequential定义的 self.net改为
self.net = [ nn.Dense(64,activation='relu'),nn.Dense(32,activation='relu') ]
会有什么影响
答： list没有forward函数，需要自己补充具体运算到forward中。

'''