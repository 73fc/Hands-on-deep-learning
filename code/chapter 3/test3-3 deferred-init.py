from mxnet import init, nd
from mxnet.gluon import nn
#-----------------------------------构建初始化类----------------------------------
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print("deferred ",name)

net = nn.Sequential()
net.add(nn.Dense(256,activation='relu'),
        nn.Dense(10))

net.initialize(init=MyInit())
'''
写错成以下形式
net.initialize(init=MyInit)
报错信息:AssertionError: initializer must be of string type
可理解为：变成了编译形的代码？

同时此时没有任何输出信息，表示并没有初始化
原因： 还不清楚网络的具体形状，不明确输入规模，无法初始化
'''
# 加入数据调用后，进行了初始化
x = nd.random.uniform(shape=(2,20))
y = net(x)


# 此时网络已经确定了形状，所以在此初始化时则会直接初始化。
net.initialize(init=MyInit(),force_reinit=True)

# 创建网络时指定层次形状，则可直接初始化
net = nn.Sequential()
net.add( nn.Dense(256,in_units=20,activation='relu'),
        nn.Dense(10,in_units=256))
net.initialize(init=MyInit())

#-------------------------------------作业----------------------------------
'''
如果下次使用网络时改变输入数据的大小，会发生什么？
答： 输入形状与模型不匹配，无法计算。如果需要使用，则要重新创建模型
'''