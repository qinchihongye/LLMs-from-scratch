#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @File    : 3.33.3Attending to different parts of the input with self-attention.py
# @Date    : 2024/12/19
# @Author  : mengzhichao
# @Version : 1.0
# @Desc    :


"""一种无需可训练权重的简单自注意力机制"""

import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
print(f"inputs.shape: {inputs.shape}")
query = inputs[1]  # 第二个 input x^2,journey

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

print(attn_scores_2)


# 点积
res = 0.

for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]

print(res)
print(torch.dot(inputs[0], query))
