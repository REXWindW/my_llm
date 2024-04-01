import os
import tiktoken
import torch
from model import GPT,Model_args

checkpoint_save_dir = './checkpoints'
device = 'cuda'
device_type = 'cuda'
dtype = 'bfloat16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# generate参数
top_k = 200
tempreture = 0.5 # 一般都先设置1，想要更random一点就往上调
start = "Sherlock Homes" # 这是最开始的输入
num_samples = 1 # sample几次
max_new_tokens = 128

# 读checkpoint
print(f"load checkpoint from {checkpoint_save_dir}")
ckpt_path = os.path.join(checkpoint_save_dir,'checkpoint.pt') # 读取checkpoint路径
checkpoint = torch.load(ckpt_path, map_location=device)
args = checkpoint['model_args']
model = GPT(Model_args(**args))
# 读取权重
# for k,v in checkpoint.items():
#     print(k)
state_dict = checkpoint['model']

# 这里nanogpt的作者说resume的时候有bug，一些参数会加上前缀'_orig_mod'
unwanted_prefix = '_orig_mod'
for k,v in list(state_dict.items()): # 遍历dict去除key中不要的前缀
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        # 截取key后半段
model.load_state_dict(state_dict)
 
model.eval()
model.to(device)

enc = tiktoken.get_encoding("gpt2")# gpt2 tokenizer
decode = lambda x:enc.decode(x)
encode = lambda x:enc.encode(x,allowed_special={"<|endoftext|>"}) 
'''
表示文本结束的特殊token，在tokenization后由开发者手动加入
在千问的文档里有个解释https://github.com/QwenLM/Qwen/blob/main/tokenization_note_zh.md
如果不加这段，<|endoftext|>会被tokenize成
ids:[1350, 9639, 91, 8691, 723, 427, 91, 82598]
tokens: [b'print', b'("<', b'|', b'endo', b'ft', b'ext', b'|', b'>")']
我们希望的情况是
ids: [1350, 445, 151643, 899]
tokens: [b'print', b'("', '<|endoftext|>', b'")']
我自己在data/test_tiktoken.py中试验了一下，
如果不增加这一参数，直接进行encode的话会直接报错
'''

start_ids = encode(start)
# x = torch.tensor(start_ids,dtype=torch.long,device=device)[None,...]
# [None,...]增加一个维度，后面...保持不变,将一维张量变成二维张量
# 或者使用unsqueeze应该也能实现
x = torch.tensor(start_ids,dtype=torch.long,device=device).unsqueeze(0)

ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# 这里显然不用再scaler了，因为不计算梯度
# 开始generate
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x,max_new_tokens,top_k=top_k,tempreture=tempreture)
            print(decode(y[0].tolist()))
            print("----------")