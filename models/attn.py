import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
"""
    scale在full attention时候是None,scale是不同的针对qk值得缩放比例，如果没有的话就使用默认的
    mask_flag false
    output_attention false
    attention dropout 0.5
"""
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        # b,l还是和之前一样，h：8，s这里应该也是seq_len,e:64,d：64都是一样的shape，从下面attention layer中找
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        # le*es调整完之后是
        # 从直觉上看 batch_size,head肯定不会变，q*k.transpose,以乘法匹配两者相似度
        # score.shape = [batch_size,head,seq_len,seq_len]
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                # attn_mask.shape = [batch_size,1,seq_len,seq_len]
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            #所以这里是对不同head分别masked么，这里和网上代码有点区别
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # A.shape = [batch_size,head,seq_len,seq_len]， attention 数值
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # V.shape = [batch_size,seq_len,head,d_key], 相当于是固定bh，这里为了和之前的v一样blhd所以结果这样已知了
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # 是因为在forward里面先transpose交换了顺序,D和E是 = d_model/n_head 64
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        # 这一步相当于沿着L_Q方向进行了复制
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        #torch.randint(upper_boound,(size)) upper bound并不包含在内，所以这里生成了一个最大值为L_k-1,shape为[L_Q,sample_k]
        # sample_k = U_part,通过index_sample实现对L_Q*L_K部分的采样
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q

        # 这里test的shape是torch.Size([32, 8, 96, 96, 25, 64])，而k_sample:torch.Size([32, 8, 96, 25, 64]是不一样的
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        #test = K_expand[:, :, :, index_sample, :]

        #         = Q（B,H,L_K.-1,D)*K(B,H,L_Q,D,n_sample) = （B,H,L_K.-1,n_sample)
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        # M(B,H,L_K)，m_top:[B,H,n_sample]
        # m_top返回的是index！！！
        # max[0]返回的就是最大的值，max[1]返回的是相应的index
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        # Q_reduce[batch_size,head,n_sample,d_model/n_head]
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        # Q_reduce[batch_size,head,n_sample,seq_len]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    #这里是对v的更新
    def _get_initial_context(self, V, L_Q):
        # mask维度和原本维度保持不变
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    # 主要是添加了prob mask和output attention，其他的计算主要是维度方面的不同，output attention是attention*v
    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        # u的取值是对原本的长度取了对数
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        #u和原本的值进行比较取较小的一个，正常情况下U_part和u是同一个
        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn

"""
    query_in:[b,s,l]->[b,s,l] 如果不单独设置d_key和n_head的话
    out: [b,s,l]
    在外面 encoder forward时候获得了query,key，value的输入
"""
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        # 如果设定了d_key那么就按照设定的d_key，同理d_value，不然的话就是按照计算得来d_model整除n_head,
        # d_key,d_value =64
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        # 这三个默认情况下是一致的，是为了数据预处理
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        # 最后再把通过attention的数据给还原回原本的shape，这里等价于nn.Linear(d_model d_model),linear是对最后一个dim操作
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        # L:seq_len,S:seq_len qkv的shape都是[batch_size,seq_len,d_model]
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # 所以这三者形状还是一致的,所以到这里的shape就变为了[batch_size,seq_len,head,d_key or d_value]
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # 这里attn为啥是none啊
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
