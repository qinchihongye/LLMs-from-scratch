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


# 简单归一化
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# softmax 归一化
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# pytorch中的 softmax
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())


# 计算 z^2
query = inputs[1] # 第二个 input x^2,journey

context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

print(context_vec_2)



"""计算所有注意力权重和上下文向量。"""

# step1
attn_scores = torch.empty(6, 6)

# 自行实现
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)

attn_scores = inputs @ inputs.T
print(attn_scores)

# step2
attn_weights = torch.softmax(attn_scores, dim=1)
print(attn_weights)

# step3
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)


print(context_vec_2)
print(all_context_vecs[1])