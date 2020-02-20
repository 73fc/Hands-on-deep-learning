import d2lzh as d2l
from mxnet import nd

#---------------------------多输入通道卷积核-------------------------------
def corr2d_multi_in(X,K):
    return nd.add_n( *[ d2l.corr2d(x,k) for x,k in zip(X,K) ] )

X = nd.array([ [ [0,1,2],[3,4,5], [6,7,8] ], [ [1,2,3],[4,5,6],[7,8,9] ] ])
K = nd.array( [ [ [0,1],[2,3] ],  [[1,2], [3,4]] ] )

corr2d_multi_in(X,K)

#---------------------------多输入输出通道卷积核-------------------------------
def corr2d_multi_in_out(X,K):
    return nd.stack( *[corr2d_multi_in(X,k) for k in K] )
#K = nd.array( [K, K + 1, K + 2] ) 为什么这样不行？
K = nd.stack(K, K + 1, K + 2)
K.shape

corr2d_multi_in_out(X,K)

#---------------------------1X1卷积核-------------------------------
'''
1x1卷积的作用与全连接等价
'''

def corr2d_multi_in_out_1X1(X,K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w ))
    K = K.reshape((c_o,c_i))
    Y = nd.dot(K , X)
    return Y.reshape((c_o, h, w))

X = nd.random.uniform(shape=(3,3,3))
K = nd.random.uniform(shape=(2,3,1,1))
Y1 = corr2d_multi_in_out_1X1(X,K)
Y2 = corr2d_multi_in_out(X,K)

(Y1- Y2).norm().asscalar() < 1e-6

#---------------------------练习-------------------------------
'''
1. 假设输入形状为 c_i * h * w 卷积核为 c_o * c_i * k_h * k_w,填充为 (p_h, p_w), 步幅为(s_h,s_w) 
问进行一次前向运算分别需要多少次乘法核加法？
答： 一次卷积有  m_1 = (k_h * k_w)次乘法 以及 a_1 (k_h*k_w - 1)次加法
    在一个输出通道在一个输入通道上需要进行:  
    c_1 = (floor( (h - k_h - 2 * p_h)/s_h ) + 1) * (floor( (w - k_w - 2 * p_w)/s_w ) + 1)次卷积运算
    相应的就会有C_1 * m_1次乘法，C_1 * a_1次加法
    所以在一个输出通道上 会进行 c_i * (C_1 * m_1) 次乘法,  c_i * (c_1 * a_1) + c_1 * (c_i - 1) + c_1 次加法
    总共 c_o 个输出通道， 所以会进行 c_o * ( c_i * C_1 * m_1 ) 次乘法
         c_o * (c_i * c_1 * a_1 + c_1 * (c_i - 1) + c_1) 次 加法
'''

'''
2.翻倍输入通道c_i和输出通道c_o会增加多少运算?翻倍填充呢？
答：由1的结果可知，翻倍输入输出通道数会使得运算量变成原来的4倍
翻倍填充增加的运算量不确定，因为由floor函数在，有可能不会增加运算量，最多增加 4倍
'''

'''
3. 如果卷积核的高和宽都为 1 能减少多少计算？
答：将k_h=k_w=1带入上式可得 在一次卷积不会有加法运算，即 m_1 = 1, a_1 = 0，
则共有 c_o * ( c_i * C_1  ) 次乘法， c_o * ( c_1 * (c_i - 1) + c_1) 次加法
'''

'''
4.本节中最后一个例子的结果 Y1和Y2完全一致吗？ 原因是什么？
答: 应该完全一致，因为计算的数据是一样的，且一一对应。
'''
'''
5.窗口不为 1x1时，如何用矩阵乘法实现卷积计算?
答： 使用扩展矩阵， 例如
C = [ c1,c2,c3;c4,c5,c6;c7,c8,c9]  与 K = [k1,k2;k3, k4] 进行卷积，可将 C扩展为
C'=[ c1, c2, c4,c5;c2, c3, c5, c6; c4, c5, c7, c8; c5, c6, c8, c9]
K'=[k1;k2,k3;k4]
'''
