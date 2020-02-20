from mxnet import gluon, nd
from mxnet.gluon import nn
#------------------------自定义无参数的网络层--------------------------
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer,self).__init__(**kwargs)
    
    def forward(self,x):
        return x - x.mean()

layer = CenteredLayer()
layer(nd.array([1,2,3,4,5,6,7]))
## 运用到复杂网络中
net = nn.Sequential()
net.add( nn.Dense(256,activation='relu'),
        CenteredLayer() )

net.initialize()
y = net(nd.random.uniform(shape=(4,8)))
y.mean().asscalar()

#------------------------自定义带参数的网络层--------------------------
# ParameterDict类以及parameter类
params = gluon.ParameterDict()
params.get('param2',shape=(2,3))
params
'''
提问: 这里生成的参数类型为<class 'numpy.float32'>，与之前似乎不同，为什么？
'''
# 自定义层
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super(MyDense,self).__init__(**kwargs)
        self.weight = self.params.get('weight',shape=(in_units,units))
        self.bias = self.params.get('bias',shape=(units,))
    def forward(self,x):
        return nd.relu(nd.dot(x,self.weight.data()) + self.bias.data())

dense = MyDense(units=3,in_units=5)
dense.params

dense.initialize()
dense(nd.random.uniform(shape=(2,5)))
'''
吐槽： 输出居然和书上一模一样，果然是伪随机
'''

# 用自定义层构建模型
net = nn.Sequential()
net.add(MyDense(8,in_units=64),
        MyDense(1,8))
net.initialize()
net(nd.random.uniform(shape=(2,64)))
'''
吐槽： 还是和书上一样
'''

#-------------------------练习------------------------------------
'''
自定义一个层，并用其做一次前向计算
答：见上面的代码
'''
