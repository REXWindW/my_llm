import os
import numpy as np
import torch
import torch.nn as nn
import math
from model import Model_args,GPT
import time

# 模型参数
block_size = 128 # 窗口大小GPT2为1024
batch_size = 32 # 暂定，之后再看显存占用
n_layer = 12
n_head = 6
n_embed = 768
bias = False
dropout = 0.0
dataset_path = './data/sherlock'
init_from = 'scratch' # 'scratch' or 'resume' # 从头训练还是继续
checkpoint_save_dir = './checkpoints'
eval_iters = 200
eval_interval = 2000 # 每n步eval和保存checkpoint一次
# 学习率衰减
learning_rate = 6e-4
warmup_iters = 2000
lr_decay_iters = 8000
min_lr = 6e-5
# 优化器参数
max_iters = 6000 # 训练多少个iter
weight_decay = 1e-1
betas = (0.9,0.95)
grad_clip = 1.0 # 梯度裁剪
# system
device = 'cuda'
device_type = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# 检查cuda是否支持bfloat16数据类型

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# torch.amp.autocast混合精度

# dataloader
data_dir = os.path.join(dataset_path)
def get_batch(split):
    # nanogpt作者说，memmap每个batch都要用一次，这样才不会内存泄漏
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data)-block_size,(batch_size,)) # 
    # torch.randint(a, b, (size,))即在（a,b）范围内生成size个随机数
    x = torch.stack([torch.from_numpy((data[i:i+block_size].astype(np.int64))) for i in ix]) # 根据ix从data里面取x,y
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size].astype(np.int64))) for i in ix])
    # torch.stack(inputs, dim=0),dim为拼接的新的维度

    x,y = x.pin_memory().to(device,non_blocking=True),y.pin_memory().to(device,non_blocking=True)
    # pin_memory()将张量锁定在内存中，non_blocking=True数据传输是非阻塞的，不会阻塞当前线程
    return x,y


model_args = dict(n_layer=n_layer, n_head=n_head, n_embed=n_embed, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

iter_num = 0 # resume的话会覆盖掉0
best_val_loss = 1e9

assert init_from == 'scratch' or init_from == 'resume'
if init_from == 'scratch': 
    print("从头训练模型")
    model_args['vocab_size'] = 50304 # gpt2 tokenizer词表大小
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

model.to(device)
optimizer = model.configure_optimizers(weight_decay,learning_rate,betas,device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None# 这时候checkpoint已经读好了，给他清空一下

# nanogpt还有个torch.compile的优化，我这里暂时先不做了

def estimate_loss():
    model.eval() # eval不计算梯度
    out = {}
    for split in ['train','val']:
        # 这里是训练集和验证集都算一下loss
        # 我发现nanogpt中很多传参都用dict的方式
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # print(f"now_eval in {k}")
            X,Y = get_batch(split)
            with ctx:
                _,loss = model(X,Y) # x,targets
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # 退出时回到train的模式
    return out


# nanogpt使用cos做learning rate的下降
def get_lr(now_iter):
    if(now_iter<warmup_iters):#(1)warmup阶段，线性上升
        return learning_rate*now_iter/warmup_iters
    elif(now_iter>lr_decay_iters):#(2)超过decay，到min了
        return min_lr
    else:# (3)在warmup和decay之间，用cos做lr衰减
        rate = (now_iter-warmup_iters)/(lr_decay_iters-warmup_iters)
        # 计算所占比例(0,1)
        return min_lr + 0.5*(1.0+math.cos(math.pi*rate)) * (learning_rate-min_lr)
    
# 训练代码
X,Y = get_batch('train')
t_before = time.time()

while(True):
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr # 设置学习率
    
    if iter_num>0 and iter_num % eval_interval == 0:
        # eval
        loss_dict = estimate_loss()
        print(f"当前进行{iter_num}个iter,train_loss:{loss_dict['train']},val_loss{loss_dict['val']}")
        best_val_loss = min(loss_dict['val'],best_val_loss)
        # save checkpoint
        checkpoint = {
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict,
            'model_args': model_args,
            'iter_num':iter_num,
            'best_val_loss':best_val_loss
        }
        torch.save(checkpoint,os.path.join(checkpoint_save_dir,'checkpoint.pt'))
        print(f"checkpoint保存在{checkpoint_save_dir}/checkpoint.pt")
    
    with ctx:
        logits,loss = model(X,Y)
        print(f"iter:{iter_num},loss:{loss.item()}")
        scaler.scale(loss).backward()
        # 用scaler，scale loss(FP16)，backward得到scaled的梯度(FP16)
    if grad_clip >0.0:
        scaler.unscale_(optimizer) # unscale梯度回fp32
        nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
        # 梯度进行裁剪，以防止梯度爆炸
    scaler.step(optimizer) # 用scaler执行optimizer.step()功能
    scaler.update() # scaler factor更新
    """
    scaler的使用，找到一篇知乎的文章https://zhuanlan.zhihu.com/p/348554267
    
    之前用了混合精度，但是把FP32到FP16时可能会溢出，所以需要乘上系数控制范围。

    GradScaler的工作就是在反向传播前给 loss 乘一个 scale factor，
    之后反向传播得到的梯度都乘了相同的 scale factor。
    并且为了不影响学习率，在梯度更新前将梯度unscale。
    步骤如下：
        维护一个 FP32 数值精度模型的副本
        在每个iteration
            拷贝并且转换成 FP16 模型
            前向传播（FP16 的模型参数）
            loss 乘 scale factor
            反向传播（FP16 的模型参数和参数梯度）
            参数梯度乘 1/scale factor
            利用 FP16 的梯度更新 FP32 的模型参数
    """
    optimizer.zero_grad(set_to_none=True) # 释放内存

    t_after = time.time()
    dt = t_after-t_before
    t_before = t_after

    iter_num += 1
    if iter_num > max_iters:
        break