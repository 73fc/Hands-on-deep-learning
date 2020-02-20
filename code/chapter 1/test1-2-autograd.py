import mxnet.ndarray as nd
import mxnet.autograd as ag

#------------------------------举例------------------------------------
##创建变量
x = nd.arange(4).reshape((4,1))
nd.arange(4)
x
##存储梯度 attach_grad
x.attach_grad()

##记录求梯度的计算，record()
y = nd.arange(3)
with ag.record():
    y = 2*nd.dot(x.T,x)
## 求梯度，只对标量求梯度，如果y不是标量，则会把y内的变量全部求和再求梯度。
y.backward()

## 验证梯度是否正确
assert(x.grad - 4*x).norm().asscalar() == 0
x.grad

## 默认使用预测模式？，调用record 以后默认为训练模式？
print(ag.is_training())
with ag.record():
    print(ag.is_training())

#------------------------------对控制流求导------------------------------------
##定义一个有控制流的函数(记录具体的运算路径再求导？)

def fun(a):
    b = a * 2
    while b.norm().asscalar() < 1000:
        b = b * 2
    if  b.sum().asscalar() > 0:
        c = b
    else:
        c = 100 * b
    return c

##求函数求导
a = nd.random.normal(shape=1)
a.attach_grad()
with ag.record():
    c = fun(a)
c.backward()

a.grad == c/a


## 作业 一
a2 = nd.random.normal(shape=(2,1))
a2.attach_grad()
with ag.record():
    c2 = fun(a2)
c2.backward()

a2.grad 
a2.grad == c2/a2

## 作业 二
def function(a):
    if 10 < a :
        a = a + 10
    else:
        a = a - 10
    return a

a4 = nd.arange(1)
a4[:] = 100
a4.attach_grad()
with ag.record():
    c4 = function(a4)
c4.backward()
a4.grad


