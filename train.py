import os
import numpy as np
import torch
from model import Model_args,GPT

# 模型参数
block_size = 256
batch_size = 2 # 暂定，之后再看显存占用
n_layer = 6
n_head = 6
n_embd = 384
bias = False
dropout = 0.0
# 训练参数
init_from = 'scratch' # 'scratch' or 'resume' # 从头训练还是继续
checkpoint_save_dir = 'checkpoints'
eval_iters = 200
# 优化器参数
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
betas = (0.9,0.95)
grad_clip = 1.0 # 梯度裁剪

# system
device = 'cuda'
device_type = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# 检查cuda是否支持bfloat16数据类型

# dataloader
data_dir = os.path.join('data')
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data)-block_size,(batch_size,)) # 
    # torch.randint(a, b, (size,))即在（a,b）范围内生成size个随机数
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64) for i in ix)]) # 根据ix从data里面取x,y
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64) for i in ix)])
    # torch.stack(inputs, dim=0),dim为拼接的新的维度

    x,y = x.pin_memory(device,non_block=True),y.pin_memory(device,non_block=True)
    # pin_memory()将张量锁定在内存中，non_blocking=True数据传输是非阻塞的，不会阻塞当前线程
    return x,y


model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

assert init_from == 'scratch' or init_from == 'resume'
if init_from == 'scratch': 
    print("从头训练模型")
    model_args.vocab_size = 50304 # gpt2 tokenizer词表大小
    # 这里直接使用GPT-2的词表，在prepare.py中，调用tiktoken.get_encoding('gpt2')来tokenize
    gpt_args = Model_args(**model_args)
    model = GPT(gpt_args) # 创建模型


elif init_from == 'resume': # 继续训练
    print("继续训练模型")
    ckpt_path = os.path.join(checkpoint_save_dir,'checkpoint.pt') # 读取checkpoint路径
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']# 从checkpoint里面读取模型参数
    for k in ['n_layer', 'n_head', 'n_embed', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gpt_args = Model_args(**model_args)
    model = GPT(gpt_args)
    state_dict = checkpoint['model'] # 模型权重
    model.load_state_dict(state_dict)

    iter_num = checkpoint['iter_num'] # 迭代器步数
    best_val_loss = checkpoint['best_val_loss']

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
# 优化：混合精度训练，大部分使用float16，少部分用float32

optimizer = model.configure_optimizers(weight_decay,learning_rate,betas,device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None# 这时候checkpoint已经读好了，给他清空一下

# nanogpt还有个torch.compile的优化，我这里暂时先不做了

def estimate_loss():
    model.eval() # eval不计算梯度
    for split in ['train','val']:
        # 这里是训练集和验证集都算一下loss
        out = {}
        losses = []
        # 我发现nanogpt中很多传参都用dict的方式
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            _,loss = model(X,Y) # x,targets
            losses.append(loss)
        out[split] = losses.mean()
    model.train() # 退出时回到train的模式
    return out

