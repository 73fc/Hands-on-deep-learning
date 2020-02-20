from mxnet import autograd,nd
from mxnet.gluon import nn
#-----------------------------------卷积操作--------------------------------
## 卷积函数
def corr2d( X, K):
    h,w = K.shape
    Y = nd.zeros( (X.shape[0] - h + 1, X.shape[1] - w + 1) )
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[ i : i+ h, j : j + w ] * K).sum()
    return Y

X = nd.array( [ [0,1,2],[3,4,5],[6,7,8] ] )
K = nd.array( [ [0,1],[2,3] ] )
corr2d( X, K)

## 构建卷积类
class Conv2D(nn.Block):
    def __init__(self, kernel_size ,**kwargs):
        super(Conv2D,self).__init__(**kwargs)
        self.weight = self.params.get('weight',shape=kernel_size)
        self.bias = self.params.get('bias',shape=(1,))
    
    def forward(self,X):
        return corr2d(X, self.weight.data()) + self.bias.data()

## 测试之边缘检测
X = nd.ones((6,8))
X[:,2:6] = 0

K = nd.array([[1,-1]])
Y = corr2d(X,K)
Y

#-------------------------------------训练卷积的参数-----------------------------------
# Gluon实现的 卷积核
con2d = nn.Conv2D( 1, kernel_size=(1,2))
con2d.initialize()

# 数据预处理
X = X.reshape((1,1,6,8))
Y = Y.reshape((1,1,6,7))
# 训练
lr = 3e-2
for i in range(10):
    with autograd.record():
        y_hat = con2d(X)
        l = (y_hat - Y) ** 2
    l.backward()
    con2d.weight.data()[:] -= lr *con2d.weight.grad()
    #con2d.bias.data()[:] -= lr * con2d.bias.grad()  
    print(" batch: %d, loss:%.3f"%(i,l.sum().asscalar()))
    
con2d.weight.data().reshape((1,2))
'''
问？ 为什么加了bias以后训练效果变差了
'''

#------------------------------------练习-----------------------------------------
# 1.设计用于检测水平方向以及对角线方向边缘的卷积核K
## 水平方向
X = nd.ones((6,8))
X[2:4,:] = 0
X
K = nd.array([ [1],[-1] ])
Y = corr2d(X,K)
Y

## 对角线方向
X = nd.ones((6,8))
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if i <= j:
            X[i,j] = 0
X
    
K = nd.array( [ [1,-1],[-1,1] ])
Y = corr2d(X,K)
Y

# 2. 用自己构造的 Conv2D类进行自动求梯度会有什么问题？

cc = Conv2D(K.shape)
cc.initialize()
with autograd.record():
    y_hat = cc(X)
    l = (y_hat - Y) ** 2
l.backward()

'''
报错信息
mxnet.base.MXNetError: [08:11:37] C:\Jenkins\workspace\mxnet-tag\mxnet\src\imperative\imperative.cc:295: Check failed: !AGInfo::IsNone(*i): Cannot differentiate node because it is not in a computational graph. You need to set is_recording to true or use autograd.record() to save computational graphs for backward. If you want to differentiate the same graph twice,
#you need to pass retain_graph=True to backward.
'''

class Conv2D2(nn.Block):
    def __init__(self, kernel_size ,**kwargs):
        super(Conv2D2,self).__init__(**kwargs)
        self.weight = self.params.get('weight',shape=kernel_size)
        self.bias = self.params.get('bias',shape=(1,))
    
    def forward(self,X):
        return nd.Convolution(data=X, weight=self.weight.data(),bias=self.bias.data())


cc = Conv2D2(K.shape)
cc.initialize()
y_hat = cc(X)
with autograd.record():
    y_hat = cc(X)
    l = (y_hat - Y) ** 2
l.backward()
'''
暂时未解决 
'''

#3. 如何通过变化输入核核的数组将互相关运算表示成一个矩阵乘法？
# 答： reshape + nd.dot

#4. 如何构造一个全连接层来进行物体边缘检测？
'''
暂时未解决 
'''