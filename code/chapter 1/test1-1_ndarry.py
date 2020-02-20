from mxnet import nd

x = nd.arange(12)
x

x.shape
x.size
x.reshape((3,4))
x
x = x.reshape((3,4))

#------------------------------创建------------------------------------
#创建0矩阵
nd.zeros((2,3,4))
nd.zeros((3,4))

# 用列表创建
y = nd.array([[2,1,4,3],[1,2,3,4],[4,3,2,1]] )
y

#创建满足正态分布的矩阵
nd.random.normal(0,1,shape=(3,3))

#------------------------------运算------------------------------------
## 加法 (形状相同)
x + y

## 按位乘法(形状相同)
x * y

##按位除法(形状相同)
x/y

##按位指数运算
y.exp()
##转置
y.T
## 矩阵乘法 dot
nd.dot(x,y.T)

##拼接
nd.concat(x,y,dim=0)
nd.concat(x,y,dim=1)

##逻辑判断
x == y
x < y

##求和
x.sum()

##L2范数，nd变量变python标量
x.norm().asscalar()


#------------------------------广播------------------------------------
A = nd.arange(3).reshape((3,1))
B = nd.arange(3).reshape((1,2))
A,B

# 加
A + B

#------------------------------索引------------------------------------
#取区间
x
x[1:3]
#改数据
x[1,2] = 7
#区间（批量）改数据
x[1:2,:] = 3
x

#------------------------------内存开销------------------------------------
## 内存的转换,创建了中间变量，再将Y指向新内存地址
before = id(y)
y = y + x
id(y) == before

## 锁定了目标地址，先创建中间地址保留结果，再复制到最终地点
z = y.zeros_like()
before = id(z)
z[:] = x + y
id(z) == before

## 指定输出位置
nd.elemwise_add(x,y,out=z)
id(z) == before

## 减少有标记的标量，从而减小空间
before = id(x)
x += y
id(x) == before

#------------------------------ndarray与numpy互换------------------------------------
## array（）  nd-->np
import numpy as np

p = np.ones((2,3))
d = nd.array(p)
d

## asnumpy() nd-->np
d.asnumpy()
type(p)
type(d)
