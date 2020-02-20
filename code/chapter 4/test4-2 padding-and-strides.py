from mxnet import nd
from mxnet.gluon import nn

#-----------------------------padding测试-------------------------------
def comp_con2d(conv2d, X):
    conv2d.initialize()
    X = X.reshape( (1,1) + X.shape )
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

# 高和宽相同的情况
cc = nn.Conv2D(1, kernel_size=3,strides=1, padding=1 )
X = nd.random.uniform(shape=(8,8))
comp_con2d(cc,X).shape

# 高和宽不同
c2 = nn.Conv2D(1,kernel_size=(5,3),padding=(2,1))
X = nd.random.normal(shape=(8,8))
comp_con2d(c2,X).shape

#----------------------------stride测试------------------------------------
# 长宽方向步调一致
c3 = nn.Conv2D(1,kernel_size=3,padding=1, strides=2)
comp_con2d(c3, X).shape

# 长宽方向步调不一致
c4 = nn.Conv2D(1,kernel_size=(3,5), padding=(0,1),strides=(3,3) )
comp_con2d(c4,X).shape



#----------------------------练习------------------------------------
'''
1.本节最后一个例子通过计算公式来计算输出形状，结果是否与实验一致？
答： 高： floor[(8 - 3)/3] + 1 = 1 + 1 = 2
     宽： floor[(8 - 5 + 1*2)/4] + 1 = 1 + 1 = 2 
     综上所述： 一致

2.尝试其他填充和步幅组合：
答： 已经尝试， 上述计算中最大的可操作空间计算 floor函数， 
即在取整结果一致的情况下，有很多步骤和填充方法可以尝试。
'''