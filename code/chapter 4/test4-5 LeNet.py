import d2lzh as d2l
import mxnet
from mxnet import init, autograd, nd, gluon
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn
import time
#--------------------------------构建模型------------------------------------

net = nn.Sequential()
net.add( nn.Conv2D(channels=6,kernel_size=5,activation='sigmoid'),
         nn.MaxPool2D(pool_size=2,strides=2),
         nn.Conv2D(channels=16,kernel_size=5,activation='sigmoid'),
         nn.MaxPool2D(pool_size=2,strides=2),
         nn.Dense(120,activation='sigmoid'),
         nn.Dense(84,activation='sigmoid'),
         nn.Dense(10))
#测试模型
X = nd.random.uniform(shape=(1,1,28,28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name,'output shape:\t',X.shape)

#---------------------------------获取数据---------------------------------
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

#---------------------------------模型训练---------------------------------
# 尝试启动GPU
def try_gpu():
    try:
        ctx = mxnet.gpu()
        _ = nd.ones( (1,), ctx=ctx)
    except mxnet.base.MXNetError:
        ctx = mxnet.cpu()
    return ctx

ctx = try_gpu()
ctx
# 准确率 in  ctx
def evaluate_accuracy(net, data_iter,  ctx):
    acc_sum, n = nd.array([0],ctx=ctx), 0
    for X,y in data_iter:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y ).sum()
        n += y.size
    return acc_sum.asscalar()/n

#训练函数
def train_ch5(net, train_iter, test_iter, trainer, ctx, batch_size, num_epochs):
    print('train on :',ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum, n ,start = 0,0,0,time.time()
        for X,y in train_iter:
            X,y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat,y).sum()
            l.backward()
            trainer.step(batch_size)
            train_l_sum += l.asscalar()
            y = y.astype('float32')
            n += y.size
            train_acc_sum += (y_hat.argmax(axis=1) == y ).sum().asscalar()
        test_acc = evaluate_accuracy(net,test_iter,ctx)
        print("epoch:%d,loss:%.3f, train_acc %.3f test_acc %.3f  time %.3f" % (epoch + 1, train_l_sum/n, train_acc_sum/n,test_acc, time.time() - start) )

lr, num_epochs = 0.8,5
net.initialize(force_reinit=True,ctx=ctx, init = init.Xavier())
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})

train_ch5(net,train_iter,test_iter,trainer,ctx,batch_size,num_epochs)

#---------------------------------练习---------------------------------
'''
1.调整模型参数，训练一个更优的模型
'''
