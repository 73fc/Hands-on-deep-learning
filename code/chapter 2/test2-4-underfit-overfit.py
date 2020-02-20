# 多项式拟合实现
#--------------------------------------构造数据集--------------------------------------------
import d2lzh as d2l
from mxnet import autograd, gluon, nd
from mxnet.gluon import loss as gloss, data as gdata, nn

## y = 1.2x - 3.4x^2 + 5.6x^3 + 5 + 随机误差
n_train, n_test, true_w,true_b = 100, 100, [ 1.2, -3.4, 5.6], 5
features = nd.random.normal(shape=(n_train + n_test,1))
poly_features = nd.concat( features, nd.power(features,2), nd.power(features,3))
true_w = nd.array(true_w).reshape( len(true_w), 1)
labels = nd.dot( poly_features, true_w) + true_b
labels += nd.random.normal(scale=0.01,shape=labels.shape)

features[:2], poly_features[:2], labels[:2]

#--------------------------------------定义数据的图像显示函数--------------------------------------------
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
            legend=None,figsize=(3.5,2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals,y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy( x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)

#--------------------------------------模型设计--------------------------------------------
num_epochs, loss = 100, gloss.L2Loss()

def fit_and_plot(train_features, test_featurs, train_labels, test_labels):
    #构建模型并初始化
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    #构造训练器
    batch_size = min(10,train_labels.shape[0])
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features,train_labels), batch_size,shuffle=True)
    trainer = gluon.Trainer( net.collect_params(), 'sgd', {'learning_rate': 0.01})
    train_ls, test_ls = [], []
    #开始训练
    for _ in range(num_epochs):
        for X,y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(loss(net(train_features),train_labels).mean().asscalar())
        test_ls.append( loss(net(test_featurs), test_labels).mean().asscalar())
    print('final epoch: train loss',train_ls[-1],'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net[0].weight.data().asnumpy(),
           '\nbias:',net[0].bias.data().asnumpy())

#---------------------------------拟合实验---------------------------------------
##正常拟合
fit_and_plot( poly_features[:n_train,:],poly_features[n_train:,:],labels[:n_train],labels[n_train:] )
##线性模型拟合(简单模型导致欠拟合)
fit_and_plot(features[:n_train,:],features[n_train:,:],labels[: n_train,:], labels[n_train:,:])
##样本不足(过拟合)
fit_and_plot( poly_features[:2,:], poly_features[n_train:,:], labels[:2],labels[n_train:])

#---------------------------------拟合实验---------------------------------------
'''
1. 用一个三阶多项式拟合一个线性模型生成的数据，会有什么问题？为什么？
答： 可能会导致过拟合，因为模型复杂度较大，学习能力太强，会学到很多数据噪音。
2.在本节中用三阶多项式拟合的问题中，是否能把100给样本的训练误差的期望降到0，为什么？
答：可能，因为本节实验中，数据的噪声就是以期望为0生成的，所以只要学到了正确的模型参数，就能使得误差的期望为0
'''