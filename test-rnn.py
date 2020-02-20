from mxnet import nd

#----------------------------一个RNN单位--------------------------------------------
#创建输入与参数 隐含状态， 注意各个量的形状大小
X, W_xh = nd.random.normal(shape=(3,1)), nd.random.normal(shape=(1,4))
H, W_hh = nd.random.normal(shape=(3,4)),nd.random.normal(shape=(4,4))
#分开相乘相加
nd.dot(X,W_xh) + nd.dot(H,W_hh)
# 利用矩阵相乘的技巧
nd.dot(nd.concat(X,H,dim=1),nd.concat(W_xh,W_hh,dim=0))


#----------------------------一个RNN单位--------------------------------------------
import d2lzh as d2l
import math
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import time

#利用工具包导入数据
(corpus_indices,char_to_idx,idx_to_char,vocab_size) = d2l.load_data_jay_lyrics()
#演示转成one-hot向量
nd.one_hot(nd.array([0,2]),vocab_size)
len(corpus_indices)

#构建字符串-->onehot编码的函数，并测试
def to_onehot(X, size):
    return [nd.one_hot(x,size) for x in X.T]

X = nd.arange(10).reshape((2,5))
inputs = to_onehot(X,vocab_size)
len(inputs),inputs[0].shape

##设计网络
num_inputs = vocab_size
num_hiddens = 256
num_outputs = vocab_size

##初始化函数
def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01,shape=shape)
    #隐含层参数
    W_xh = _one((num_inputs,num_hiddens))
    W_hh = _one((num_hiddens,num_hiddens))
    b_h = _one(num_hiddens)
    #输出层参数
    W_hq = _one((num_hiddens,num_outputs))
    b_q = _one(num_outputs)
    params = [ W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params

##定义模型
def init_rnn_state(batch_size,num_hiddens):
    return (nd.zeros(shape=(batch_size,num_hiddens)),)

def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X,W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs,(H,)

### 测试
state = init_rnn_state(X.shape[0],num_hiddens)
inputs = to_onehot(X,vocab_size)
params = get_params()
outputs ,state_new = rnn(inputs,state,params)
len(outputs),outputs[0].shape, state_new[0].shape


##预测函数
def predict_rnn(prefix,num_chars,rnn,params,init_rnn_state,num_hiddens,vocab_size,idx_to_char,char_to_idx):
    state = init_rnn_state( 1, num_hiddens)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = to_onehot(nd.array([output[-1]]), vocab_size)
        (Y,state) = rnn(X,state,params)
        if t < len(prefix) - 1:
            output.append(char_to_idx[ prefix[t+1] ])
        else:
            output.append( int( Y[0].argmax(axis=1).asscalar() ) )
    return ''.join([idx_to_char[i] for i in output])


predict_rnn('不分开',10,rnn,params,init_rnn_state,num_hiddens,vocab_size,idx_to_char,char_to_idx)

##梯度裁剪
def grad_clipping(params, theta):
    norm = nd.array([0])
    for param in params:
        norm += (param.grad ** 2 ).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta/norm


def train_and_predict_rnn(rnn,get_params,init_rnn_state,num_hiddens,
                            vocab_size,corpus_indices,idx_to_char,
                            char_to_idx, is_random_iter, num_epochs, num_steps,
                            lr, clipping_theta, batch_size,pred_period,
                            pred_len,prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:
            state = init_rnn_state(batch_size,num_hiddens)
        l_sum ,n , start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps)
        for X,Y in data_iter:
            if is_random_iter:
                state = init_rnn_state(batch_size, num_hiddens)
            else:
                for s in state:
                    s.detach()
            with autograd.record():
                inputs = to_onehot(X, vocab_size)
                (outputs, state) = rnn(inputs, state, params)
                outputs = nd.concat(*outputs,dim=0)
                y = Y.T.reshape((-1,))
                l = loss(outputs,y).mean()
            l.backward()
            grad_clipping(params,clipping_theta)
            d2l.sgd(params,lr,1)
            l_sum += l.asscalar() * y.size
            n += y.size
        print("epoch %d" % epoch)
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec'% ( epoch + 1, math.exp(l_sum/n), time.time() - start ) )
            for prefix in prefixes:
                print("-",predict_rnn(prefix, pred_len, rnn, params, init_rnn_state, 
                          num_hiddens,vocab_size,idx_to_char,char_to_idx))


num_epochs, num_steps, batch_size, lr, clipping_theta = 500, 35, 128, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开','不分开','朝维']

train_and_predict_rnn(rnn,get_params,init_rnn_state,num_hiddens,
                            vocab_size,corpus_indices,idx_to_char,
                            char_to_idx, False, num_epochs, num_steps,
                            lr, clipping_theta, batch_size,pred_period,
                            pred_len,prefixes)