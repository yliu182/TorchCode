# PyTorch Learning

---

## Part 1: Common Mistakes

### 1. `torch.transpose` 需要指定维度

在实现 `SimpleLinear` 时，想计算 `y = x @ W^T + b`：

```python
# 错误 — 缺少维度参数，会报错
return torch.matmul(x, torch.transpose(self.weight)) + self.bias

# 正确 — 指定交换 dim 0 和 dim 1
return torch.matmul(x, torch.transpose(self.weight, 0, 1)) + self.bias

# 更简洁
return x @ self.weight.T + self.bias
```

### 2. `requires_grad` 与非叶子张量

初始化权重时，除法会产生新的计算图节点，导致张量变成非叶子节点，`.grad` 不会在 `backward()` 时被填充：

```python
# 错误 — 除法创建了新张量，不再是叶子节点
self.weight = torch.randn(out_features, in_features, requires_grad=True) / math.sqrt(in_features)
# RuntimeError: .grad attribute of a Tensor that is not a leaf Tensor is being accessed

# 正确 — 先算完再用 in-place 方法设置 requires_grad
self.weight = (torch.randn(out_features, in_features) / math.sqrt(in_features)).requires_grad_(True)
self.bias = torch.zeros(out_features).requires_grad_(True)
```

**原理**：`requires_grad_(True)` 是 in-place 操作，不会创建新节点，张量保持为叶子节点。

### 3. `^` 不是幂运算

在实现 Layer Norm 时，计算方差：

```python
# 错误 — ^ 在 Python 中是按位异或（XOR），不是幂运算
delta = torch.sum((x - u)^2, dim=-1, keepdim=True)

# 正确
delta = torch.sum((x - u)**2, dim=-1, keepdim=True)
# 或
delta = torch.sum(torch.pow(x - u, 2), dim=-1, keepdim=True)
```

### 4. `math.sqrt` 不能作用于张量

Layer Norm 归一化时：

```python
# 错误 — math.sqrt 只接受 Python 标量，delta 是张量
return gamma * (x - u) / math.sqrt(delta + eps) + beta

# 正确 — 用 torch.sqrt 处理张量
return gamma * (x - u) / torch.sqrt(delta + eps) + beta
```

### 5. `torch.sum` vs `torch.mean` — 反复犯错

在 Layer Norm、RMS Norm 中都犯了同样的错误——用 `torch.sum` 而非 `torch.mean`：

```python
# Layer Norm 方差 — 错误
delta = torch.sum((x - u)**2, dim=-1, keepdim=True)  # 没有除以元素数量，值偏大

# Layer Norm 方差 — 正确
delta = torch.mean((x - u)**2, dim=-1, keepdim=True)

# RMS Norm — 同样的错误
rms = (torch.sum(x ** 2, dim=-1, keepdim=True) + eps) ** 0.5   # 错误
rms = (torch.mean(x ** 2, dim=-1, keepdim=True) + eps) ** 0.5  # 正确
```

**规则**：公式中有 `1/d * Σ` 就用 `torch.mean`，别用 `torch.sum`。

### 6. Running stats 更新必须 in-place

在 Batch Normalization 中更新 running_mean / running_var：

```python
# 错误 1 — += 会多加一倍
running_mean += (1 - momentum) * running_mean + momentum * mean
# 等价于 running_mean = running_mean + (1-momentum)*running_mean + momentum*mean
#       = (2-momentum)*running_mean + momentum*mean  ← 多了一倍

# 错误 2 — = 赋值创建新张量，调用者看不到更新
running_mean = (1 - momentum) * running_mean + momentum * mean
# 函数内的局部变量指向新张量，调用者持有的还是旧张量

# 正确 — 用 [:] 切片赋值，修改原张量的内存
with torch.no_grad():
    running_mean[:] = (1 - momentum) * running_mean + momentum * mean
    running_var[:] = (1 - momentum) * running_var + momentum * var

# 或用 in-place 链式操作
with torch.no_grad():
    running_mean.mul_(1 - momentum).add_(momentum * mean)
    running_var.mul_(1 - momentum).add_(momentum * var)
```

**原理**：Python 函数内重新赋值局部变量不影响外部引用。必须用 in-place 操作修改原张量内容。

### 7. `masked_fill` 不能用乘法替代

实现 causal attention 时：

```python
# 错误思路 — 乘 0 不能屏蔽 attention
scores = scores * (1 - mask)  # 未来位置变成 0，但 softmax(0) ≠ 0，还是有概率

# 正确 — 必须用 -inf，softmax(-inf) = 0
mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))

# 也可以用加法代替 masked_fill
mask = torch.triu(torch.ones(S, S), diagonal=1)
scores = scores + mask * float('-inf')
```

### 8. `masked_fill` 的 mask 含义搞反

实现 sliding window attention 时：

```python
# 错误 — mask 中 True 表示窗口内（允许），但 masked_fill 会把 True 的位置填 -inf
mask = ...  # True = 窗口内
scores = scores.masked_fill(mask, float('-inf'))  # 反了！屏蔽了窗口内的位置

# 正确 — 取反，屏蔽窗口外的位置
scores = scores.masked_fill(~mask, float('-inf'))
```

**规则**：`masked_fill(mask, value)` 把 `mask=True` 的位置填充为 value。

### 9. Sliding window mask 函数的多个 bug

```python
# Bug 1: torch.max/torch.min 不能用于标量
left = torch.max(i - window_size, 0)    # 错误
left = max(i - window_size, 0)          # 正确 — 用 Python 内置函数

# Bug 2: mask 初始化全 1，窗口内也设为 1，等于没做
mask = torch.ones(len_seq, len_seq)     # 全 1
mask[i][left:right+1] = 1              # 还是 1，没效果
# 应该初始化为全 0
mask = torch.zeros(len_seq, len_seq)

# Bug 3: 返回时引用了函数本身而非变量
return create_mask.bool()               # 错误 — create_mask 是函数名
return mask.bool()                      # 正确

# 向量化版本（无需 for 循环）
def create_mask(len_seq, window_size):
    idx = torch.arange(len_seq)
    return (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= window_size
```

### 10. Attention 缩放因子写错

```python
# 错误 — d_k 的平方
scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 2)

# 正确 — d_k 的平方根
scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
```

### 11. `softmax` 的 dim 参数传了字符串

```python
# 错误 — dim 是整数，不是字符串
attn = torch.softmax(scores, dim='-1')

# 正确
attn = torch.softmax(scores, dim=-1)
```

### 12. Multi-head attention 中 transpose 维度选错

将 `(B, S, num_heads, d_k)` 转换为 `(B, num_heads, S, d_k)` 时：

```python
# 错误 — transpose(-2, -1) 交换最后两维，变成 (B, S, d_k, num_heads)
q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(-2, -1)

# 正确 — transpose(1, 2) 交换 dim 1 和 dim 2
q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
# (B, S, num_heads, d_k) → (B, num_heads, S, d_k)
```

### 13. `nn.Linear` 是 module，不能当矩阵用 matmul

在 GQA 实现中：

```python
# 错误 — nn.Linear 不是权重矩阵，不能直接 matmul
Q = torch.matmul(x, self.W_q)

# 正确 — nn.Linear 是可调用的 module
Q = self.W_q(x)
```

### 14. `repeat_interleave` 需要整数，整除要用 `//`

```python
# 错误 — / 返回 float，repeat_interleave 需要 int
num_repeat = self.num_heads / self.num_kv_heads

# 正确
num_repeat = self.num_heads // self.num_kv_heads
```

### 15. GQA 中 repeat 维度不对

不能在最后一维直接 repeat，那样是重复特征值而不是复制 head：

```python
# 错误 — 在 d_k 维度上 repeat，把特征值重复了
K = torch.repeat_interleave(K, num_repeat, dim=-1)

# 正确 — 先 reshape 成多头，再在 head 维度上 repeat
k = self.W_k(x).view(B, S, self.num_kv_heads, d_k).transpose(1, 2)
# (B, num_kv_heads, S, d_k)
k = k.repeat_interleave(num_repeat, dim=1)
# (B, num_heads, S, d_k)

# 或用 expand（不复制内存，更高效）
k = self.W_k(x).view(B, S, self.num_kv_heads, d_k)
k = k.unsqueeze(3).expand(-1, -1, -1, num_repeat, -1)
k = k.reshape(B, S, self.num_heads, d_k).transpose(1, 2)
```

### 16. GQA forward 最后 matmul 参数写反 + 缺少 W_o

```python
# 错误 — scores 和 attn 都不是 V
return torch.matmul(scores, attn)

# 正确 — attn weights 乘以 V，再经过输出投影
out = torch.matmul(attn, v)
out = out.transpose(1, 2).reshape(B, S, self.d_model)
return self.W_o(out)
```

---

## Part 2: Key Concepts

### Tensor Shape 索引

```python
Q = torch.randn(3, 10, 128)  # (batch, seq, d_k)

Q.shape[-1]    # 128 — 最后一个维度（整数）
Q.shape[:-1]   # torch.Size([3, 10]) — 除最后一个外的所有维度（元组）
```

`-1` 是索引取单个元素，`:-1` 是切片取多个元素。

### `torch.matmul` 行为取决于输入维度

```python
# 1D @ 1D → 标量（点积）
torch.matmul(torch.randn(3), torch.randn(3))  # tensor(1.23)

# 2D @ 2D → 矩阵乘法
torch.matmul(torch.randn(3,4), torch.randn(4,5))  # (3, 5)

# 3D @ 3D → 批量矩阵乘法（最后两维相乘）
torch.matmul(torch.randn(2,3,4), torch.randn(2,4,5))  # (2, 3, 5)
```

| 输入 | 行为 | 结果 |
|------|------|------|
| 1D @ 1D | 点积 | 标量 |
| 2D @ 2D | 矩阵乘法 | 2D |
| 1D @ 2D / 2D @ 1D | 向量×矩阵 | 1D |
| nD @ nD | 批量矩阵乘法 | nD |

### 矩阵乘法 API 对比

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)

A @ B                      # 等价于 torch.matmul，最常用
torch.mm(A, B)             # 严格 2D × 2D
torch.bmm(batch_A, batch_B)  # 严格 3D 批量
torch.einsum('ik,kj->ij', A, B)  # 最灵活
```

### `torch.einsum` 示例

```python
# 矩阵乘法
torch.einsum('ik,kj->ij', A, B)

# 点积
torch.einsum('i,i->', v, v)

# 外积
torch.einsum('i,j->ij', v, v)

# 批量矩阵乘法
torch.einsum('bik,bkj->bij', Q, K)

# Attention: Q @ K^T
torch.einsum('bqd,bkd->bqk', Q, K)
```

**规则**：重复的下标求和，保留的下标保留。

### `transpose` 与 `contiguous()`

```python
# transpose 只改 stride 元数据，不移动内存
x = x.view(B, S, num_heads, d_k).transpose(1, 2)  # (B, num_heads, S, d_k)

# view 要求内存连续，transpose 后必须先 contiguous()
out = x.transpose(1, 2).contiguous().view(B, S, -1)

# reshape 自动处理，等价于 contiguous().view()
out = x.transpose(1, 2).reshape(B, S, -1)
```

### Softmax dim 的选择

Attention scores shape 为 `(B, seq_q, seq_k)`，softmax 在 `dim=-1`（key 维度）上做：

```python
attn = torch.softmax(scores, dim=-1)  # 每个 query 对所有 key 的分数归一化为概率
output = attn @ V  # 对 V 的加权平均，权重和为 1
```

**为什么沿 key 维度**：每个 query 独立决定"关注哪些 key 多一点"，注意力权重加起来等于 1 才是有意义的加权平均。

### Self-Attention vs Cross-Attention

```python
# Self-attention: Q、K、V 来自同一序列
q = self.W_q(x)  # x: (B, S, D)
k = self.W_k(x)
v = self.W_v(x)
# seq_q == seq_k

# Cross-attention: Q 来自 decoder，K/V 来自 encoder
q = self.W_q(decoder_out)   # (B, seq_q, D) — 目标语言
k = self.W_k(encoder_out)   # (B, seq_k, D) — 源语言
v = self.W_v(encoder_out)
# seq_q != seq_k，attention score 形状为 (B, seq_q, seq_k)
```

### GQA vs MHA vs MQA

```
MHA (8 heads):        每个 Q head 有独立的 K/V head
Q:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8
K:  K1 K2 K3 K4 K5 K6 K7 K8

GQA (8 Q heads, 2 KV groups):  多个 Q head 共享一组 K/V
Q:  Q1 Q2 Q3 Q4 | Q5 Q6 Q7 Q8
K:     K1        |     K2

MQA (1 KV group):    所有 Q head 共享同一组 K/V
Q:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8
K:           K1
```

GQA 减少 KV head → KV cache 更小 → 推理更快。LLaMA 2 70B、Gemma、Mistral 使用。

### Causal Mask（因果遮罩）

```python
# 每个位置只能 attend 到自己和之前的位置
mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
# [[F, T, T, T],
#  [F, F, T, T],
#  [F, F, F, T],
#  [F, F, F, F]]
scores = scores.masked_fill(mask, float('-inf'))
```

### KV Cache

```python
# Prefill: 处理完整 prompt，生成 cache
out, cache = model(prompt_tokens, cache=None)

# Decode: 逐 token 生成，只投影新 token
for _ in range(max_tokens):
    out, cache = model(next_token, cache=cache)
    # next_token: (B, 1, D)，只算 1 个新 token
    # cache 中的 K/V 通过 torch.cat 拼接新的 k/v

# forward 中的 cache 拼接
if cache is not None:
    k = torch.cat([cache[0], k], dim=2)  # (B, H, S_past+S_new, d_k)
    v = torch.cat([cache[1], v], dim=2)
```

### Cross-Entropy Loss 与 LogSumExp 技巧

```python
def cross_entropy_loss(logits, targets):
    # LogSumExp: log(Σ exp(x)) = max + log(Σ exp(x - max))
    # 减去 max 防止 exp 溢出，数学上等价
    max_logits = logits.max(dim=-1, keepdim=True).values
    log_sum_exp = max_logits.squeeze(-1) + torch.log(
        torch.exp(logits - max_logits).sum(dim=-1)
    )

    # fancy indexing: 从每行取出目标类别的 logit
    correct_logits = logits[torch.arange(logits.shape[0]), targets]

    # CE = -log(exp(x_y) / Σexp(x_j)) = -x_y + log(Σexp(x_j))
    return (log_sum_exp - correct_logits).mean()
```

**数学推导**：
```
log(Σ exp(x_j))
= log(Σ exp(x_j - m) · exp(m))       # m = max(x)
= m + log(Σ exp(x_j - m))            # 提出 exp(m)
# exp(x_j - m) 最大值为 1，不会溢出
```

### SwiGLU MLP

```python
class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff)   # 门控
        self.up_proj = nn.Linear(d_model, d_ff)      # 内容
        self.down_proj = nn.Linear(d_ff, d_model)    # 投影回来

    def forward(self, x):
        # SiLU(gate) * up = 门控机制，gate 控制信息流
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# SiLU(x) = x * sigmoid(x)，也叫 Swish
# LLaMA、Mistral、PaLM 使用，优于传统 ReLU/GELU FFN
```

### GPT-2 Block (Pre-Norm)

```python
class GPT2Block(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        # ... attention weights and MLP ...
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.causal_self_attention(self.ln1(x))  # 残差 + attention
        x = x + self.mlp(self.ln2(x))                     # 残差 + MLP
        return x
```

**Pre-Norm**：LayerNorm 在 attention/MLP **之前**（而非之后），训练更稳定。

### Linear Attention — O(SD²) 替代 O(S²D)

```python
def linear_attention(Q, K, V, feature_map=None):
    # 核技巧：找 φ 使得 sim(q,k) ≈ φ(q)·φ(k)^T
    # elu(x)+1 保证非负，模拟 softmax 的概率性质
    Q = F.elu(Q) + 1
    K = F.elu(K) + 1

    # 关键：改变计算顺序
    # 标准: (Q @ K^T) @ V  →  中间产物 (S, S)  → O(S²D)
    # 线性: Q @ (K^T @ V)  →  中间产物 (D, D)  → O(SD²)
    KV = torch.matmul(K.transpose(-2, -1), V)  # (B, D, D)
    out = torch.matmul(Q, KV)                   # (B, S, D)

    # 归一化：模拟 softmax 的分母 Σ exp(q·k)
    Z = torch.matmul(Q, K.transpose(-2, -1).sum(dim=-1, keepdim=True))
    return out / (Z + 1e-6)
```

### Batch Normalization

```python
def my_batch_norm(x, gamma, beta, running_mean, running_var,
                  eps=1e-5, momentum=0.1, training=True):
    if training:
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)  # 用 N 而非 N-1
        with torch.no_grad():  # running stats 是 buffer，不需要梯度
            running_mean[:] = (1 - momentum) * running_mean + momentum * mean
            running_var[:] = (1 - momentum) * running_var + momentum * var
    else:
        mean = running_mean  # 推理时用累积的统计量
        var = running_var

    return gamma * (x - mean) / torch.sqrt(var + eps) + beta
```

### Layer Norm vs RMS Norm

```python
# Layer Norm: 减均值，除标准差
def my_layer_norm(x, gamma, beta, eps=1e-5):
    u = torch.mean(x, dim=-1, keepdim=True)
    var = torch.mean((x - u)**2, dim=-1, keepdim=True)
    return gamma * (x - u) / torch.sqrt(var + eps) + beta

# RMS Norm: 不减均值，只除 RMS（更简单，LLaMA/Gemma 使用）
def rms_norm(x, weight, eps=1e-6):
    rms = (torch.mean(x ** 2, dim=-1, keepdim=True) + eps) ** 0.5
    return (x / rms) * weight
```

