import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    自注意力机制使用了prob attention，cross attention还是使用了传统transformer
    d_ff:Dimension of fcn (defaults to 2048)
    return [batch_size,label+pred_len,d_model]
"""
class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        # 这两次卷积结束后没有改变形状
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # layer norm也是最后一个维度
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        # cross 是encoder out, cross.shape:[batch_size,seq_len/(2*n),d_model] n为encoder中蒸馏的次数
        # x.shape:[batch_size,label+pred,d_model]
        # key和value的维数是一样的，query里面的seq_len可以是不一样的，最后返回的是V，是和query的shape一致的
        # 所以跨注意力机制是，查询了当前值和encoder注意力之间的值
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        # 沿着label+pred_len的方向上面进行卷积操作
        # 同时这里有个残差
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        # elementwise的相加
        return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x