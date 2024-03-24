import torch
import torch.nn as nn

@dataclass
# 模型参数
class model_args:
    block_size: int = 1024
    # vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    # 我这里用
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0 # 默认不dropout
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

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

class flash_att(nn.Module):
    # 参考NanoGPT
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
        # 等价于nn.Dropout(p=self.dropout)
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
        return self.att_dropout(self.proj(y))
        

class MLP(nn.Module):
    # MLP部分参考llama MLP结构
    def __init__(self,args):
        super.__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.up_proj = nn.Linear(args.n_embed, 4*args.n_embed, bias = args.bias)
        self.down_proj = nn.Linear(4*args.n_embed, args.n_embed, bias = args.bias)
        # 使用relu
        self.act_func = nn.functional.relu
        # 学习llama增加一个门控
        self.gate = nn.Linear(args.n_embed, 4*args.n_embed, bias = args.bias)

    def forward(self, x):
        # llama代码把MLP输入X切片成slice，我这里就不切片了
        gate_proj = self.gate(x)
        x = self.up_proj(x)

        # llama中的代码：
        # intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
        # nanogpt的
        # x = self.act_func(x)
        # 发现这里区别主要在，nanogpt对upproj的x使用激活函数，llama则是对gate使用

        x = self.act_func(gate_proj)*x # 和门控gate按照对应位置相乘
        x = self.down_proj(x)
        return self.dropout(x)
    
class Block(nn.Module):
    # 之后用来堆叠的block
    def __init__(self, args):
        super().__init__()
        self.norm = RMS_Norm(args.n_embed)
        self.attn = flash_att(args)
        self.mlp = MLP(args)

    def forward(self,x):
        # 使用pre norm
        x = x + self.attn(self.norm(x))# residual
        return x + self.mlp(self.norm(x)) # 残差链接

class GPT(nn.Module):
    # llama和GPT2的缝合怪
    def __init__(self,args):
        super().__init__()
        
        # 我打算是直接使用transformers库的tokenizer和embedding
        