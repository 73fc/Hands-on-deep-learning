from mxnet import nd
from mxnet.gluon import nn
#---------------------最大池化层与平均池化层----------------------------
def pool2d(X, pool_size, mode='max'):
    p_h,p_w = pool_size
    Y = nd.zeros( ( X.shape[0] - p_h + 1, X.shape[1] - p_w + 1 ) )
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i,j] = X[i: i + p_h, j : j + p_w].max()
            elif mode == 'avg':
                Y[i,j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

# 数据
X = nd.array([ [0,1,2],[3,4,5],[6,7,8]])
# 最大池化
pool2d(X,(2,2))
#平均池化
pool2d(X,(2,2),'avg')

#------------------------------填充与步幅-------------------------------
#数据
X = nd.arange(16).reshape((1,1,4,4))
X
#nn的池化层，默认步幅为(p_h,p_w) 互不相干
pool = nn.MaxPool2D(3)
pool(X)  
#指定步幅和填充
pool2 = nn.MaxPool2D(3,padding=1,strides=2)
pool2(X)
# h 和 w 可以不同，分布指定即可
pool3 = nn.MaxPool2D((2,3), padding = (1,2), strides=(2,3))
pool3(X)

#------------------------------多通道-------------------------------
#数据
X = nd.concat(X, X + 1, dim=1)
X

#池化
pool4 = nn.MaxPool2D(3,padding=1,strides=2)
pool4(X)

#------------------------------练习-------------------------------
'''
1.设输入形状为 c * h * w, 使用 p_h,p_w大小的池化窗口，(p_h,p_w)大小的填充和(s_h,s_w)的步幅，问计算复杂度
答：  书中填充的大小与池化窗口大小一致，觉得可能有错误，使用将填充大小改为(f_h,f_h)来讨论，
如果有问题可带回原数据。
     根据题意得： 需要进行的池化次数为 (floor((h + 2*f_h - p_h )/s_h) + 1) * ( floor((w + 2*f_w - p_w )/s_w) + 1)
'''
'''
2. 最大池化与平均池化在作用上可能的区别有那些？
答： 最大池化希望抓住窗口内最突出的特征，平均池化希望抓住窗口内个元素的共性特征 
    一种可能的情况是最大池化抓住的是图像中的特殊物体， 而平均池化抓住的是图像背景
'''
'''
3.你觉得最小池化层这个想法有没有意义？
答： 没有，因为稍加变化就可将最小池化变成最大池化来解决。所以只需要留最大池化即可。
'''
