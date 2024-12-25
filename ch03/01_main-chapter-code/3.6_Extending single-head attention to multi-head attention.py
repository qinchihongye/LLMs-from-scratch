
import torch
from torch import nn

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
batch = torch.stack((inputs, inputs), dim=0)


class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec


"""3.6.1 Stacking multiple single-head attention layers"""


class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


torch.manual_seed(123)

context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


"""3.6.2 Implementing multi-head attention with weight splits"""

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """  
        Parameters:
        -------------------------------------------------------------------------------------------------------------------
        d_in:                type: int            desc:  输入特征的维度               
        d_out:               type: int            desc:  输出特征的维度       
        context_length:      type: int            desc:  上下文长度，即输入序列的长度  
        dropout:             type: float          desc:  Dropout概率，用于防止过拟合。             
        num_heads:           type: int            desc:  注意力头的数量            
        qkv_bias:            type: boolean        desc:  是否在查询（Query）、键（Key）、值（Value）的线性变换中使用偏置项，默认为False            
        
        """
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        # 分别定义query\key\value的线性变换层，将输入维度d_in 映射到输出维度d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 定义一个线性变换层，用于将多个注意力头的输出组合成最终的输出
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        # 定义一个Dropout层，用于在训练过程中随机丢弃一部分神经元，以防止过拟合。
        self.dropout = nn.Dropout(dropout)
        # 注册因果掩码
        # self.register_buffer将一个张量注册为模型的缓冲区，这样它不会被当作模型参数进行优化，但会随着模型一起保存和加载。
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),diagonal=1) # 生成一个上三角矩阵，对角线以上的元素为1，其余为0。这个矩阵用作因果掩码，确保在计算注意力时，每个位置只能关注到它之前的位置。
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape # b批次大小，num_tokens序列长度，d_in输入特征维度
        
        # 计算QKV
        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # 重塑张量形状
        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # 转置向量
        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力分数
        # 。keys.transpose(2, 3) 将键的形状从 (b, num_heads, num_tokens, head_dim) 转置为 (b, num_heads, head_dim, num_tokens)，以便进行矩阵乘法
        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # 应用因果掩码
        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # 将因果掩码转换为布尔类型，并截取到当前序列长度。

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf) # 使用掩码将注意力分数中不应该关注的位置设置为负无穷，这样在后续的Softmax操作中，这些位置的权重会趋近于0。
        
        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) # 除以 keys.shape[-1]**0.5 是为了缩放注意力分数，防止梯度消失或爆炸
        attn_weights = self.dropout(attn_weights) #  对注意力权重应用Dropout

        # 计算上下文向量
        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # 合并注意力头
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) # 将多个注意力头的输出合并成一个张量，形状为 (b, num_tokens, d_out)。
        # 应用输出投影
        context_vec = self.out_proj(context_vec) # optional projection 对合并后的上下文向量进行线性变换，得到最终的输出

        return context_vec

torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

print('context_vecs.\n',context_vecs)
print("context_vecs.shape:\n", context_vecs.shape)


# (b, num_heads, num_tokens, head_dim) = (1, 2, 3, 4)
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],

                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])

print(a @ a.transpose(2, 3))




first_head = a[0, 0, :, :]
first_res = first_head @ first_head.T
print("First head:\n", first_res)

second_head = a[0, 1, :, :]
second_res = second_head @ second_head.T
print("\nSecond head:\n", second_res)


print(a.shape)