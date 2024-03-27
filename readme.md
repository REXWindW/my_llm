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
