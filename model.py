import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import inspect

# 模型参数
from dataclasses import dataclass
@dataclass
class Model_args:
    block_size: int = 1024 # 传入的最大大小
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0 # 默认不dropout
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class RMS_Norm(nn.Module):
    # 参考llama使用RMS Norm
    def __init__(self,hidden_size,eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        # 引入eps避免分母为0

    def forward(self,hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        sqrt_pow_mean = torch.sqrt(hidden_states.pow(2).mean(-1, keepdim = True))
        # 这里计算L2范式/n后开根，详见RMS Norm的定义
        return self.weight * hidden_states/(sqrt_pow_mean+self.eps)

class flash_att(nn.Module):
    # 参考NanoGPT
    def __init__(self,args):
        super().__init__()
        # qkv合到一个Linear里面去
        self.qkv_atten = nn.Linear(args.n_embed,3*args.n_embed,bias = args.bias)
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
        self.c_proj = nn.Linear(self.n_embed,self.n_embed, bias = args.bias)

    def forward(self, x):
        B,T,C = x.shape
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
        return self.att_dropout(self.c_proj(y))
        

class MLP(nn.Module):
    # MLP部分参考llama MLP结构
    def __init__(self,args):
        super().__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.up_proj = nn.Linear(args.n_embed, 4*args.n_embed, bias = args.bias)
        self.down_c_proj = nn.Linear(4*args.n_embed, args.n_embed, bias = args.bias)
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
        x = self.down_c_proj(x)
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
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(args.vocab_size, args.n_embed),
            # 获取token_embed
            wpe = nn.Embedding(args.block_size, args.n_embed),
            # 使用一组可学习的位置编码pos_embed
            drop = nn.Dropout(args.dropout),
            h = nn.ModuleList([Block(args) for i in range(args.n_layer)]),
            norm = RMS_Norm(args.n_embed)
        ))

        self.lm_head = nn.Linear(args.n_embed, args.vocab_size,bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # 这里不是简简单单的赋值，而是wte和lm_head共享参数
        # lm_head (n_embed,vocab_size)相当于从词向量到token的预测
        # wte ()
        
        self.apply(self._init_weights) # 初始化权重
        n_sum = 0
        # 正态分布初始化attention的投影层和MLP的下采样
        for pname,p in self.named_parameters():
            n_sum = n_sum + p.numel() # 顺带统计一下参数
            if pname.endswith('c_proj.weight'):# c_proj是上下文感知的投影层
                torch.nn.init.normal_(p,mean=0.0, std=0.02/math.sqrt(2*args.n_layer))

        print(f"模型参数量：{n_sum}")
    
    def _init_weights(self,module):# 初始化先行层和embedding
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets = None): # targets是训练时传入的目标，用来计算交叉熵loss
        device = idx.device
        B,T = idx.size()
        pos = torch.arange(0,T,dtype=torch.long,device=device) # 位置

        # embedding
        token_embed = self.transformer.wte(idx) # (B,T,n_embed)
        pos_embed = self.transformer.wpe(pos)# (t,n_embed)
        # 位置embed可学习

        x = self.transformer.drop(token_embed+pos_embed) # 合并token和pos
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.norm(x)

        # 经过lm_head
        # target= True 表示模型正在训练阶段，需要回传loss
        # logits取最后一个（-1）即生成出来的东西，这样和目标的一个token维度相同，才好计算损失

        if targets is not None:
            logits = self.lm_head(x)
            # 用-1取最后一维度个，把前面的t丢掉(t,vocab_size)->(vocab_size)
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index=-1) # 交叉熵损失
        else: # generate时使用
            logits = self.lm_head(x)
            loss = None

        return logits,loss

    def configure_optimizers(self,weight_decay,learning_rate,betas,device_type):
        # 建立一个从参数名到参数的dict
        param_dict = {pn:p for pn,p in self.named_parameters()}
        # 再去掉不用计算梯度的部分
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad }

        # weight decay
        # 对二维的参数使用weight decay，其他不用，这样分成两组
        decay_params = [p for pn,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for pn,p in param_dict.items() if p.dim() < 2]
        # dict.items()是返回一个key和value元组的list [(k1,v1),(k2,v2)]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # 统计一下decay和不decay的参数量
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"使用weight decay的参数量为{num_decay},不使用weight decay的参数量为{num_nodecay}")

        # 这段是建立一个AdamW优化器，看版本是否支持fused融合
        # 判断Adam的参数字典中是否包含fused，如果有，把它添加到extra args中
        fused_avail = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # inspect.signature(fn).parameters返回参数list
        use_fused = fused_avail and device_type == 'cuda' # 并且要有gpu
        if use_fused:
            print("AdamW optimiser use fused!")
        extra_args = {'fused':True} if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups,lr = learning_rate,betas = betas,**extra_args)
        # betas:计算梯度以及梯度平方的运行平均值的系数
        # ** 用于将一个字典解包成关键字参数传递给函数
        
        return optimizer

    def generate(self, idx, max_generate_tokens, tempreture=1.0, top_k=None):
        # topp，topk和tempreture的概念
        # max_generate_tokens为生成的新tokens的最大数量
        for _ in range(max_generate_tokens):
            idx = idx if idx.shape[1] <= self.args.block_size else idx[:,-self.args.block_size:]
            # 如果大于传入的最大大小则截取后面一段
            # 其实这里我有点不懂，如果idx长度不足blocksize，是哪一步给他填充到blocksize大小的呢？
            logits, _ = self(idx)
            logits = logits[:,-1,:]/tempreture #(B,T,C)取最后一个即新生成的
            # tempreture更高，生成的随机性更高
            # 从这里能知道，是softmax的性质决定的，指数函数小的时候变化小，不同token的probs差距会被减少，随机性就强了

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf') # 忽略topk名以后的token

            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # 按照probs概率选一个
            idx = torch.cat((idx, idx_next), dim=1)

        return idx