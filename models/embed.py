import torch
import torch.nn as nn
import torch.nn.functional as F

import math
"""
    return [1,seq_len,d_model]
"""
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        # position.shape是[5000,1]
        position = torch.arange(0, max_len).float().unsqueeze(1)
        #div_term.shape = [256]
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        #这里时候是[5000,512],这里执行了一个广播机制
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe最后shape是[1,5000,512]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.shape[32,96,7]就是在dataloader里面输出的那个[batch_size,seq_len,7],7是feature的数目
        # 这里打印出来的输出值，也就是return的值的shape是[1,96,512],其实就是对上面那个pe的数值进行了裁剪
        # test = self.pe[:, :x.size(1)]
        return self.pe[:, :x.size(1)]

"""
    c_in:7不确定这是不是输入的通道的数目
    将输入的通道数目转化为了，d_model的数目
    输出结果是[batch_size,seq_len,d_model]
"""
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        # 这里只是1维卷积而不是1*1卷积，kernel=3，padding=1这样卷积完之后长度是不变的，因为kernel导致-2，padding左右各+1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        # 对层进行了相应的初始化，token都是需要嵌入的
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        # 从原本的x[batch_size,seq_len,c_in]变为了卷积结束的[batch_size,seq_len,d_model]
        test = x
        # 这里我理解的形状并不改变，因为permute之后又transpose了过来
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

"""
    模仿了position encoding，将其中的encoding部分作为了不可训练的网络
    return [batch_size,seq_len,d_model]
"""
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

"""
    timefix时候使用的是这个，
    实际就是使用了一层线性神经网络来学习这个，类似于教程里面的EmbeddingsWithLearnedPositionalEncoding
    input = [batch_size,seq_len,time_feature] time feature是在dataloader里面看到的4
    return = [batch_size,seq_len,d_model]
"""
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        test = x
        test1 = self.embed(x)
        return self.embed(x)


class NoTimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super(NoTimeEmbedding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        batch_size, seq_len, time_featue = x.shape
        # 这里因为不是葱x继承过来的，所以需要cuda
        zero = torch.zeros(batch_size, seq_len, self.d_model).cuda()
        return zero

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # fixed执行的是temporalEmbedding，而time feature执行的是相当于对年月日进行了编码这个就不仔细研究了
        #self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        if embed_type == 'timeF':
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        elif embed_type == 'none':
            self.temporal_embedding = NoTimeEmbedding(d_model=d_model)
        else:
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # embedding是对三个量都分别嵌入后进行相加，之后需要对batch_size方向执行了增广操作
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)