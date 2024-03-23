import torch
import torch.nn as nn

class RMS_Norm(nn.Module):
    # 参考llama使用RMS Norm
    def __init__(self,hidden_size,eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self._eps = eps
        # 引入eps避免分母为0

    def forward(self,hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        sqrt_pow_mean = torch.sqrt(hidden_states.pow(2).mean(-1, keep_dim = True))
        # 这里计算L2范式/n后开根，详见RMS Norm的定义
        return self.weight * hidden_states/(sqrt_pow_mean+self.eps)

class Attention(nn.Module):
    # 注意力
    def __init__(self,args):
        super().__init__()
        # qkv合到一个Linear里面去
        self.qkv_atten = nn.Linear(3*args.n_embed,args.n_embed,bias = args.bias)
        # 记得有一篇论文说head_size要等于seq_length才合理
        self.n_head = args.n_head
        self.n_embed = args.n_embed
        # 计算一下head_size
        assert args.n_embed % args.n_head == 0
        self.head_size = args.n_embed//args.n_head
        # dropout
        self.dropout = args.dropout # 这里是存布尔值，参数dropout概率，generate时设置为0即可
        self.att_dropout = nn.Dropout(self.dropout)
        # projection layer
        self.proj = nn.Linear(self.n_embed,self.n_embed, bias = args.bias)

    def forward(self, x):
        B,T,C = x.shape()
        # x的尺寸：(B,T,C)
        q, k, v = self.qkv_atten(x).split(self.n_embed,dim = 2) # B,T,C
        
        q = q.view(B,T,self.n_head,self.head_size).transpose(1,2)
        # (B,T,C) -> (B,T,n_head,head_size) -> (B,n_head,T,head_size)
        k = k.view(B,T,self.n_head,self.head_size).transpose(1,2)
        v = v.view(B,T,self.n_head,self.head_size).transpose(1,2)

        # 使用torch封装好的flash attention
        y = nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=None,
                                                       dropout_p = self.dropout if self.training else 0, 
                                                       is_causal=True)
        # 训练时dropout
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # 之后要transpose一下让它变成(B,T,nh,hs)
        y = y.transpose(1,2)# (B,T,nh,hs)
        # .contiguous()方法会返回一个张量，保证了其在内存中的连续性
        y = y.contiguous().view(B,T,C) # (B,T,C)
        
        # 输出时经过投影层后dropout
        y = self.att_dropout()
    