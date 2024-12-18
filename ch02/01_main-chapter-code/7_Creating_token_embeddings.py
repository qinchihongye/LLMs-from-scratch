#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @File    : 7_Creating_token_embeddings.py
# @Date    : 2024/12/18
# @Author  : mengzhichao
# @Version : 1.0
# @Desc    :

import torch
import tiktoken

input_ids = torch.tensor([2, 3, 5, 1])


#为了简单起见，假设我们有一个只有6个单词的小词汇表，并且我们希望创建大小为3的嵌入
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

print(embedding_layer(torch.tensor([3])))

print(embedding_layer(input_ids))
