# 项目日志

## 3-24

- 完成模型结构
- 在MLP部分做了修改，增加一层门控
- 使用RMS-Norm，有说法是RMS-Norm好在不改变词向量方向

## 3-26

- 完成optimizer，generate函数

## 3-27

- tiktoken用法[ChatGPT丨使用tiktoken计算tokens - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/629776230)
- data/prepare.py使用tiktoken处理input.txt的莎士比亚数据集
- train.py中读取checkpoint的部分代码，如果是resume的话，从checkpoint['model_args']继承之前训练的参数，从checkpoint['num_iter']读取之前训练到哪里
- getbatch部分，torch.randint随机从中采样，并且通过torch.stack在0维拼接成batch

## 3-28

- estimate_loss在训练集和测试机上都抽取eval_iter个来计算loss
- 发现nanogpt中，很多传参都是通过dict的方式传的，如out['train']和out['eval']分别保存训练集和验证集上的loss

## 3-29

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