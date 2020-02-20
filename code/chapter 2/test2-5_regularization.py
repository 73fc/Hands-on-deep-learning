import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

#----------------------------------------------构造数据集-------------------------------------------
n_train,n_test ,num_inputs = 20, 100, 200
true_w ,true_b = nd.ones((num_inputs,1)) * 0.01, 0.05

features = nd.random_normal(shape=(n_train + n_test, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random_normal(scale=0.01,shape=labels.shape)
train_features, test_features=features[: n_train,:], features[n_train:,:]
train_labels, test_labels = labels[:n_train], labels[n_train:]

#----------------------------------------------构建模型-------------------------------------------
#初始化参数
def init_params():
    W = nd.random.normal(scale=1,shape=(num_inputs,1))
    b = nd.zeros((1,))
    W.attach_grad()
    b.attach_grad()
    return [W,b]

#L2 正则化项
def l2_penalty(w):
    return (w**2).sum()/2

##训练且测试
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features,train_labels),batch_size=batch_size,shuffle=True)


def fit_and_plot(lamda):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X,w,b),y) + l2_penalty(w) * lamda 
            l.backward()
            d2l.sgd([w,b],lr,batch_size)
        train_ls.append(loss(net( train_features,w,b), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features,w,b), test_labels).mean().asscalar())
    d2l.semilogy( range( 1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls,['train','test'] )
    print('L2 norm of w:',w.norm().asscalar())
#fit_and_plot(lamda=0)

#fit_and_plot(lamda=3)


#----------------------------------------------gluon实现-------------------------------------------
def fit_and_plot(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    train_w = gluon.Trainer(net.collect_params('.*weight'),'sgd',{'learning_rate':lr,'wd':wd})
    train_b = gluon.Trainer(net.collect_params('.*bias'),'sgd',{'learning_rate':lr})
    data_iter = gdata.DataLoader(gdata.ArrayDataset(train_features,train_labels),batch_size,shuffle=True)
    train_ls, test_ls = [],[]

    for _ in range(num_epochs):
        for X,y in data_iter:
            with autograd.record():
                l = loss(net(X),y)
            l.backward()
            train_w.step(batch_size)
            train_b.step(batch_size)
        train_ls.append(loss(net(train_features),train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features),test_labels).mean().asscalar())
    d2l.semilogy(range(1,num_epochs+1), train_ls, 'epochs','loss', 
                    range(1,num_epochs+1),test_ls,['train','test'])
    print("L2 norm of w:",net[0].weight.data().norm().asscalar())

fit_and_plot(wd=3)

#----------------------------------------------练习题-------------------------------------------
'''
1.还有什么对付过拟合的方法？
答: k折，多模型组合
2.正则化与贝叶斯统计中哪个重要概念相对应？
答：从贝叶斯的角度来看，正则化等价于对模型参数引入 先验分布 。
3.调节超参数，观察并分析实验结果。
'''
