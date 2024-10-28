# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial


class Spatial_Attention_layer(nn.Module):
    """
    compute spatial attention scores
    """

    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(
            torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE)
        )
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(
            torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE)
        )
        self.Vs = nn.Parameter(
            torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE)
        )

    def forward(self, x):
        """
        基于空间维度（图中节点）的注意力机制
        input
            x: (batch_size, N, F_in, T)   批次，节点数，特征数，时间步长
        return:
            output                        批次，节点数，节点数
        """

        # x.shape:[32, 307, 1, 72] 批次，节点数，特征数，时间步长  
        # (B, N, F, T)(T) -> (B, N, F)(F, T) -> (B, N ,T)       # 将空间维度（节点数）放在第二个维度上，方便后续对空间维度进行操作
        lhs = torch.matmul(
            torch.matmul(x, self.W1), self.W2
        )

        # (F)(B, N, F, T) -> (B, N, T) -> (B, T, N)
        rhs = torch.matmul(self.W3, x).transpose(
            -1, -2
        )

        # (B, N, T)(B, T, N) -> (B, N, N)                       # 生成空间维度（节点数）的相关性矩阵(B, N, N) 捕捉空间维度的依赖关系
        product = torch.matmul(lhs, rhs)

        S = torch.matmul(
            self.Vs, torch.sigmoid(product + self.bs)
        )
        
        # 节点维度的归一化  (N, N)(B, N, N) -> (B, N, N)
        S_normalized = F.softmax(S, dim=1)

        return S_normalized


class cheb_conv_withSAt(nn.Module):
    """
    K-order chebyshev graph convolution
    """

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        """
        使用切比雪夫多项式执行图卷积操作，捕获图中的空间依赖关系（基于空间注意力相关矩阵）
        input:
            K: int                          # 切比雪夫多项式的阶数
            in_channles                     # 输入通道数              
            out_channels                    # 输出通道数，将其带到其他维度数中，捕获特征维度的信息
        """
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList(
            [
                nn.Parameter(
                    torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)
                )
                for _ in range(K)
            ]
        )

    def forward(self, x, spatial_attention):
        """
        模型能够利用图的结构信息（通过切比雪夫多项式）和注意力机制（通过注意力矩阵）来更新每个节点的特征表示，从而捕获复杂的空间依赖关系

        input:
             x: (batch_size, N, F_in, T)       # 图信号（原始矩阵）
             spatial_attention                 # 空间注意力矩阵，用于调整节点间的关系（这就是为什么 V 在函数外，因为这里要用注意力相关性矩阵）    
        
        return: 
            output:(batch_size, N, F_out, T)
        """

        # 批次大小，节点数，通道数，时间步长
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        # 时间步循环
        for time_step in range(num_of_timesteps):
            # 提取当前时间步的图信号（input x  节点的矩阵）
            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            # 初始化输出张量
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(
                self.DEVICE
            )  # (b, N, F_out)

            # 对于每个阶数 K
            for k in range(self.K):
                # 获取第 K 阶切比雪夫多项式（公式 3）
                T_k = self.cheb_polynomials[k]  # (N,N)
                
                # 计算与空间注意力矩阵的乘积，通过使用空间注意力矩阵，可以动态调解节点间的关系，从而增强卷积操作的表达能力
                T_k_with_at = T_k.mul(
                    spatial_attention
                )  # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                # 目的：通过矩阵乘法，将调整后的邻接关系应用于输入信号（就是原始矩阵）
                # T_k_with_at 表示节点之间经过注意力调整后的关系
                rhs = T_k_with_at.permute(
                    0, 2, 1
                ).matmul(
                    graph_signal
                )  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                # 可学习参数
                theta_k = self.Theta[k]  # (in_channel, out_channel)

                # 将结果累加在 output 中（第 K 阶值）
                # 将结果与对应的可学习参数 theta_k 相乘
                output = output + rhs.matmul(
                    theta_k
                )  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            # 将每个时间步的输出添加到 outputs 列表中，扩展维度以适应时间维度
            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(
            torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE)
        )
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(
            torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE)
        )
        self.Ve = nn.Parameter(
            torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE)
        )

    def forward(self, x):

        """
        该是基于时间维度的注意力机制，捕捉不同时间步间的依赖关系
        """
        """
        input:
            x: (batch_size, N, F_in, T)       # 批次，节点，特征数，时间步长（模型需要捕捉这些时间点间的关系）
        
        return: 
            output:(B, T, T)                  # 
        """
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # x.shape:[32, 307, 1, 72] 批次，节点数，特征数，时间步长

        # 生成查询矩阵 Q
        # (B, N, F, T) -> (B, T, F, N)      将时间维度放在第二个维度上，方便后续对时间维度进行操作
        # (B, T, F, N)(N) -> (B, T, F)
        # (B, T, F)(F, N) -> (B, T, N)
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)

        # 生成键矩阵 K
        # (F)(B,N,F,T)->(B, N, T)
        rhs = torch.matmul(self.U3, x)

        # 乘积计算
        # (B, T, N)(B, N, T)->(B, T, T)   生成时间步间的相关性矩阵(B, T, T) 捕捉时间维度的依赖关系
        product = torch.matmul(lhs, rhs)

        # 注意力分数计算
        # (B, T, T)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be)) 

        # 归一化
        E_normalized = F.softmax(E, dim=1)

        # (B, T, T)
        return E_normalized


class cheb_conv(nn.Module):
    """
    K-order chebyshev graph convolution
    """

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        """
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        """
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList(
            [
                nn.Parameter(
                    torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)
                )
                for _ in range(K)
            ]
        )

    def forward(self, x):
        """
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        """

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(
                self.DEVICE
            )  # (b, N, F_out)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class ASTGCN_block(nn.Module):
    def __init__(
        self,
        DEVICE,
        in_channels,
        K,
        nb_chev_filter,                         # 代表切比雪夫图卷积的输出通道数，通过不同的通道数来增强模型的表达能力
        nb_time_filter,                         # 代表时间卷积的输出通道数，用于提取时间维度上的特征，帮助模型捕获时序信息；不同的时间通道数允许模型处理复杂的时间依赖关系
        time_strides,
        cheb_polynomials,
        num_of_vertices,
        num_of_timesteps,
    ):
        super(ASTGCN_block, self).__init__()
        self.TAt = Temporal_Attention_layer(
            DEVICE, in_channels, num_of_vertices, num_of_timesteps
        )
        self.SAt = Spatial_Attention_layer(
            DEVICE, in_channels, num_of_vertices, num_of_timesteps
        )
        self.cheb_conv_SAt = cheb_conv_withSAt(
            K, cheb_polynomials, in_channels, nb_chev_filter
        )
        self.time_conv = nn.Conv2d(
            nb_chev_filter,
            nb_time_filter,
            kernel_size=(1, 3),
            stride=(1, time_strides),
            padding=(0, 1),
        )
        self.residual_conv = nn.Conv2d(
            in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides)
        )
        self.ln = nn.LayerNorm(nb_time_filter)  # 需要将channel放到最后一个维度上

    def forward(self, x):

        """
        使用时间和空间注意力机制以及卷积操作，以捕捉时空数据中的复杂依赖关系
        """
        """
        input:  
            x: (batch_size, N, F_in, T)                         # 批次，节点数，特征数，时间步长
        
        return: 
            output: (batch_size, N, nb_time_filter, T)          # 批次，节点数，
        """

        # x.shape: [32, 307, 1, 72]   批次，节点数，特征数，时间步长（不同时间模式）
        # 这里和传统的 NLP 相同（还没有用到 图结构信息）
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # 时间注意力，该是基于时间维度的注意力机制，捕捉不同时间步间的依赖关系
        temporal_At = self.TAt(x)  # (b, T, T)

        # 将重塑的 input 与时间注意力矩阵相乘（思路与 QK^T * V 相同思路）
        x_TAt = torch.matmul(
            x.reshape(batch_size, -1, num_of_timesteps), temporal_At
        ).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)

        # 基于空间注意力的图卷积
        # 计算空间注意力矩阵，这里没有计算完，就计算得到空间相关矩阵，然后放到图卷积中使用
        # x_TAt.shape: [32, 307, 1, 72] 批次大小，节点数，维度数，时间步长
        spatial_At = self.SAt(x_TAt)

        # 使用空间注意力矩阵进行图卷积，捕捉图结构中空间依赖关系
        # spatial_At.shape: [64, 307, 307], x.shape:[64, 307, 1, 72]
        # spatial_gcn.shape: [64, 307, 64, 72]
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)  # (b,N,F,T)


        # 时间卷积
        """
        目的：在提取空间特征基础上，进一步捕获时间维度上的依赖关系
        这里使用时间卷积的作用：
            提取局部时间特征：时间注意力机制获取全局的时间依赖关系，而时间卷积获取部分时间特征
            降维和特征变换

        self.time_conv = nn.Conv2d(             # 卷积操作（二维卷积）
            nb_chev_filter,                     # 输入通道                    
            nb_time_filter,                     # 输出通道
            kernel_size=(1, 3),                 # 表示在时间维度上应用 1D 卷积（跨越 3 个时间步），而在节点维度上不进行卷积
            stride=(1, time_strides),           # 控制时间步的跳跃，允许提取不同的时间特征
            padding=(0, 1),                     # 确保卷积操作后的时间维度大小不变
        )

        kernel_size=(1, 3)
            1：表示在第三个维度（通常是节点维度）上，卷积核的大小为1。这意味着卷积核在这个维度上不进行跨越，只对单个特征或节点进行操作
            3：表示在第四个维度（通常是时间维度）上，卷积核的大小为3。这意味着卷积核会跨越连续的3个时间步来进行卷积操作
        
        stride=(1, time_strides)
            1：在节点维度上，步幅为 1，意味着卷积核在节点维度上逐个移动，不跳过任何节点
            time_strides：在时间维度上，步幅为 time_strides，意味着卷积核在时间维度上每次移动 time_strides 个时间步
        """
        # 沿着时间轴应用 2D 卷积以提取时间特征
        # 
        time_conv_output = self.time_conv(
            spatial_gcn.permute(0, 2, 1, 3)
        )  # (B, N, F, T)->(B, F, N, T) 用(1,3)的卷积核去做->(B, F, N, T)

        # 残差连接
        x_residual = self.residual_conv(
            x.permute(0, 2, 1, 3)
        )  # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)
        
        # 这里的残差连接是直接开始连接最后（有点搞笑） 
        x_residual = self.ln(
            F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)
        ).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)

        return x_residual


class ASTGCN_submodule(nn.Module):
    def __init__(
        self,
        DEVICE,
        nb_block,
        in_channels,
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        cheb_polynomials,
        num_for_predict,
        len_input,
        num_of_vertices,
    ):
        """
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        """

        super(ASTGCN_submodule, self).__init__()

        self.BlockList = nn.ModuleList(
            [
                ASTGCN_block(
                    DEVICE,
                    in_channels,
                    K,
                    nb_chev_filter,
                    nb_time_filter,
                    time_strides,
                    cheb_polynomials,
                    num_of_vertices,
                    num_of_timesteps=len_input,                              
                )
            ]
        )

        self.BlockList.extend(
            [
                ASTGCN_block(
                    DEVICE,
                    nb_time_filter,
                    K,
                    nb_chev_filter,
                    nb_time_filter,
                    1,
                    cheb_polynomials,
                    num_of_vertices,
                    len_input // time_strides,
                )
                for _ in range(nb_block - 1)
            ]
        )

        self.final_conv = nn.Conv2d(
            int(len_input / time_strides),
            num_for_predict,
            kernel_size=(1, nb_time_filter),
        )

        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x):
        """
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        """

        # 子模块
        # x.shape:[32, 307, 1, 72]    批次，节点数，特征维度（只有流量维度的），时间步长（不同时间模式）
        for block in self.BlockList:
            x = block(x)

        # 卷积层
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)

        return output


def make_model(
    DEVICE,
    nb_block,
    in_channels,
    K,
    nb_chev_filter,
    nb_time_filter,
    time_strides,
    adj_mx,
    num_for_predict,
    len_input,
    num_of_vertices,
):
    """
    构建基于注意力的时空卷积网络模型

    input:
        nb_block                # STGCN 块的数量
        in_channels             # 数据的维度
        K                       # 切比雪夫多项式的阶数，表示用于图卷积的邻接节点数量，图卷积可以通过切比雪夫多项式来逼近图拉普拉斯矩阵
        nb_chev_filter          # 切比雪夫滤波器的数量，表示图卷积层的输出通道数，控制图卷积后的输出特征维度 
        nb_time_filter:         # 时间卷积滤波器的数量，表示时间卷积层的输出通道数，控制时间维度卷积后的输出特征
        time_strides:           # 时间卷积的步幅，控制时间卷积时的下采样程度
        adj_mx                  # 邻接矩阵，用于表示图的结构包括节点间的连接关系
        num_for_predict:        # 预测步数，表示模型最终测试多少时间步长的数据
        len_input               # input 序列的长度，就是时间步长（72=12*2*3）
        num_of_vertices         # 图的节点数量

    return:
        model                   # 模型
        
    """ 

    # 计算缩放拉普拉斯矩阵
    L_tilde = scaled_Laplacian(adj_mx)

    # 计算切比雪夫多项式（就是公式 3）
    cheb_polynomials = [
        torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE)
        for i in cheb_polynomial(L_tilde, K)
    ]

    # 实例化模型
    model = ASTGCN_submodule(
        DEVICE,
        nb_block,
        in_channels,
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        cheb_polynomials,
        num_for_predict,
        len_input,
        num_of_vertices,
    )

    # 初始化模型参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model
