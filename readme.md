# 单卡训练的LLM

## reference
- [NanoGPT](https://github.com/karpathy/nanoGPT/tree/master)
- [Llama](https://github.com/meta-llama/llama)

## introduction

- 主要是为了自己当前阶段学习的内容做一个实践
- 代码主要参考nanogpt的基础上做的小修改，并且减少了很多功能，基本只留下了最基础的模型结构和训练代码，去掉注释应该只有300行左右
- 后续打算把自己新学到的都在上面应用一下，比如lora微调

## requirements

```
pytorch==2.0以上版本
numpy
```

## 使用方法

- 目前只支持n卡训练不支持cpu，不支持DDP
- 参照data/shakespare中的形式准备好input.txt,运行prepare.py处理出train.bin和val.bin
- 修改train.py中的dataset_path为数据集路径
- 根据自己的显存占用修改一下batch_size,n_embed和n_layers等参数

## 项目日志

### 3-24

- 完成模型结构
- 在MLP部分做了修改，增加一层门控
- 使用RMS-Norm，有说法是RMS-Norm好在不改变词向量方向

### 3-26

- 完成optimizer，generate函数

### 3-27

- tiktoken用法[ChatGPT丨使用tiktoken计算tokens - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/629776230)
- data/prepare.py使用tiktoken处理input.txt的莎士比亚数据集
- train.py中读取checkpoint的部分代码，如果是resume的话，从checkpoint['model_args']继承之前训练的参数，从checkpoint['num_iter']读取之前训练到哪里
- getbatch部分，torch.randint随机从中采样，并且通过torch.stack在0维拼接成batch

### 3-28

- estimate_loss在训练集和测试机上都抽取eval_iter个来计算loss
- 发现nanogpt中，很多传参都是通过dict的方式传的，如out['train']和out['eval']分别保存训练集和验证集上的loss

### 3-29

- 写了get_lr，线性warmup，然后用cos来做学习率衰减，直到min_lr停止衰减
- 混合精度训练，torch.amp.autocast混合精度，配合torch.cuda.amp.GradScaler对梯度进行缩放
```
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
```
- 完成train loop代码


### 3-30 debug
- 缺少__init__方法报错
```
https://blog.csdn.net/wuShiJingZuo/article/details/134903071
gpt_args = Model_args(**model_args)
TypeError: Model_args() takes no arguments

在Model_args声明前增加了一行@dataclass 装饰器自动为该类生成了 __init__、__repr__、__eq__ 等方法
```
- dict.items()是返回一个key和value元组的list [(k1,v1),(k2,v2)]
- optimizer使用fused=True报错，只有在cuda上面的tensor才可以使用fused
```
optimizer = model.configure_optimizers(weight_decay,learning_rate,betas,device_type)
这里报错RuntimeError: `fused=True` requires all the params to be CUDA, floating point Tensor
我在前面加了一句model.to(device)把model移动到gpu上。nanogpt源码没有这句，不知道怎么跑通的
```
- Wqkv三个和一，shape应该是(n_embed,3*n_embed),一开始弄反了

### 3-31 debug
- 之前显存占用都很低，执行estimate_loss()时爆显存了
```
报错位置:
losses = torch.zeros(eval_iters)
......省略......
losses[k] = loss
应该改成losses[k] = loss.item()省去没必要的梯度信息
```
- 算是能够跑通train了，之后的任务大概如下
```
(1)确定一组合适单卡训练的参数(n_embed,n_layers),确定合适的训练轮次max_iters
(2)补充sample.py生成的代码
(3)后续继续追加lora微调等功能的代码
```

### 4-1 sample.py
- 在定义encoder的时候，设置了allowed_special参数
```
encode = lambda x:enc.encode(x,allowed_special={"<|endoftext|>"}) 
表示文本结束的特殊token，在tokenization后由开发者手动加入
在千问的文档里有个解释https://github.com/QwenLM/Qwen/blob/main/tokenization_note_zh.md
如果不加这段，<|endoftext|>会被分开tokenize成
ids:[1350, 9639, 91, 8691, 723, 427, 91, 82598]
tokens: [b'print', b'("<', b'|', b'endo', b'ft', b'ext', b'|', b'>")']
我们希望的情况是
ids: [1350, 445, 151643, 899]
tokens: [b'print', b'("', '<|endoftext|>', b'")']
```
- 占用4G显存，训练了6000轮
```
block_size = 128
n_layer = 10
n_head = 6
n_embed = 768
效果很差，而且有一个问题是，他会一直重复生成同一个单词好几次，就像这样
“Your sister asked asked asked at my flight.”
” he remarked.
“It looks newer have some through on German your?”
“No, it.”
“No,”
```